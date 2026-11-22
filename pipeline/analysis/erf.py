"""Campo Receptivo Efectivo (ERF) — Cap 4, eje D. Figura estrella del paper de VMamba:
Mamba tiene ERF *global* (linear-cost long-range); la CNN, *local*. Es una ventaja
ARQUITECTÓNICA de Mamba independiente de la madurez de los kernels (no depende de FPS).

Método (Luo et al. 2016, el mismo del paper VMamba):
  1. Entrada con requires_grad. Forward hasta el ÚLTIMO mapa de features espacial (B,C,H,W).
  2. Backprop del gradiente de la celda CENTRAL de ese mapa hacia la entrada.
  3. Acumular |grad de la entrada| sobre N imágenes → promediar sobre canales → heatmap.
  4. CNN → mancha central pequeña (local). Mamba → energía repartida (global).

Requiere GPU y el checkpoint del modelo.

Ejemplos:
  python pipeline/analysis/erf.py --config pipeline/configs/resnet50.yaml
  python pipeline/analysis/erf.py --config pipeline/configs/vmamba_t.yaml

Salida: _master/erf_<modelo>.npy (matriz HxW) + _master/erf_<modelo>.png (heatmap).
Genera el panel comparativo con: pipeline/analysis/erf.py --combine resnet50 vmamba_t efficientvmamba_s

Nota: el script CAPTURA automáticamente el último tensor 4D del forward y REPORTA su
nombre de módulo, shape y layout detectado. VERIFICA en el stdout que:
  - para la CNN sea un feature map plausible (p.ej. [16,2048,7,7] en resnet50),
  - para Mamba el layout (NCHW vs NHWC) esté bien detectado (VMamba suele ser NHWC:
    [16, H, W, C]). Si el layout sale mal, pásalo con --layout nhwc|nchw.
"""
import argparse
import os

import numpy as np
import torch


def _capture_last_feature_map(model, x, layout):
    """Registra hooks; devuelve (feature_tensor, name, layout) del último 4D espacial."""
    captured = {}

    def hook(mod, inp, out, name):
        t = out[0] if isinstance(out, (tuple, list)) and len(out) else out
        if torch.is_tensor(t) and t.dim() == 4:
            captured["t"] = t
            captured["name"] = name

    handles = [m.register_forward_hook(
        lambda mod, i, o, n=n: hook(mod, i, o, n)) for n, m in model.named_modules()]
    _ = model(x)
    for h in handles:
        h.remove()
    if "t" not in captured:
        raise RuntimeError("No se capturó ningún tensor 4D. Pasar --feature-module manual.")

    t = captured["t"]
    b, d1, d2, d3 = t.shape
    if layout == "auto":
        # NCHW: canales en dim1 (grande), H≈W en dims 2,3. NHWC: canales en dim3.
        layout = "nchw" if (d1 >= d3 and abs(d2 - d3) <= 1) else "nhwc"
    print(f"[erf] feature map: módulo='{captured['name']}' shape={tuple(t.shape)} "
          f"layout={layout}")
    return t, captured["name"], layout


def compute_erf(model, device, resolution, n_images, layout, data_root=None):
    model.eval().to(device)
    erf = None
    for i in range(n_images):
        if data_root:  # imágenes reales (mejor); si no, ruido gaussiano (reproducible)
            x = _load_image(data_root, i, resolution).to(device)
        else:
            x = torch.randn(1, 3, resolution, resolution, device=device)
        x.requires_grad_(True)

        feat, _, lay = _capture_last_feature_map(model, x, layout)
        layout = lay  # fija el detectado tras la 1ª imagen
        # celda central del mapa espacial, sumada sobre canales
        if layout == "nchw":
            H, W = feat.shape[2], feat.shape[3]
            center = feat[:, :, H // 2, W // 2].sum()
        else:  # nhwc
            H, W = feat.shape[1], feat.shape[2]
            center = feat[:, H // 2, W // 2, :].sum()

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        center.backward()
        g = x.grad.detach().abs().mean(dim=(0, 1))  # (res,res), promedio sobre canales
        erf = g if erf is None else erf + g
    erf = (erf / n_images).cpu().numpy()
    return erf  # ERF CRUDO (mean|grad|). La normalización de display se hace al graficar
    #           (log saturaba y escondía la concentración central de la CNN).


def _load_image(data_root, idx, resolution):
    from PIL import Image
    import torchvision.transforms as T
    files = sorted(
        os.path.join(dp, f) for dp, _, fs in os.walk(data_root)
        for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise RuntimeError(f"Sin imágenes en {data_root}")
    tf = T.Compose([T.Resize((resolution, resolution)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(files[idx % len(files)]).convert("RGB")
    return tf(img).unsqueeze(0)


def _norm_display(erf, p=99.5):
    """Normalización lineal con recorte al percentil p. A diferencia del log, PRESERVA la
    concentración central de la CNN (que es el contraste que se quiere mostrar)."""
    e = erf / (erf.max() + 1e-12)
    hi = np.percentile(e, p)
    return np.clip(e / (hi + 1e-12), 0, 1)


def save_heatmap(erf, out_png, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(_norm_display(erf), cmap="inferno")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combine(names, root):
    """Panel comparativo desde los .npy ya generados."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        erf = np.load(os.path.join(root, f"erf_{name}.npy"))
        ax.imshow(_norm_display(erf), cmap="inferno")
        ax.set_title(f"{name}\nfuera del centro: {100*(1-_central_fraction(erf)):.0f}%",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle("Campo receptivo efectivo: Mamba (global) vs CNN (local)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(root, "fig_erf_comparativo")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("Guardado:", out + ".pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dataset", default="imagenette", choices=["imagenette", "isic"])
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--layout", default="auto", choices=["auto", "nchw", "nhwc"])
    ap.add_argument("--data-root", default=None, help="imágenes reales; si se omite, ruido")
    ap.add_argument("--combine", nargs="+", default=None,
                    help="modelos a combinar en un panel (usa los .npy ya generados)")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "_master"))
    os.makedirs(root, exist_ok=True)

    if args.combine:
        combine(args.combine, root)
        return

    from train import load_config, set_determinism
    from models.registry import build_model
    cfg = load_config(args.config, args.dataset)
    set_determinism(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg["model"]["name"], cfg["num_classes"],
                        pretrained=cfg["pretrained"],
                        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"))
    name = cfg["model"]["name"]
    erf = compute_erf(model, "cuda", args.resolution, args.n_images,
                      args.layout, args.data_root)
    np.save(os.path.join(root, f"erf_{name}.npy"), erf)
    save_heatmap(erf, os.path.join(root, f"erf_{name}.png"), name)
    print(f"[erf] {name}: guardado erf_{name}.npy/.png | "
          f"energía fuera del cuadrante central 25%: "
          f"{100 * (1 - _central_fraction(erf)):.1f}%")


def _central_fraction(erf):
    """Fracción de energía del ERF en el 25% central (métrica: CNN alta, Mamba baja)."""
    H, W = erf.shape
    e = erf / (erf.sum() + 1e-12)
    h0, h1 = H // 4, 3 * H // 4
    w0, w1 = W // 4, 3 * W // 4
    return float(e[h0:h1, w0:w1].sum())


if __name__ == "__main__":
    main()

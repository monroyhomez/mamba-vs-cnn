"""Perfil radial del ERF (Cap 4, eje D — la métrica cuantitativa que SÍ discrimina).

El heatmap del ERF a la última capa no discrimina bien (todos se ven globales). El perfil
RADIAL normalizado al centro sí: mide cómo cae |grad| con la distancia al centro.
  - CNN (local): decae con la distancia (borde/centro < 1).
  - Mamba jerárquico (global): plano o creciente (borde/centro ≈ 1).
  - Mamba+conv híbrido: intermedio.

Lee los `erf_<modelo>.npy` (ERF crudo) generados por erf.py. Corre LOCAL (no necesita GPU).

Uso:
  python pipeline/analysis/erf_radial.py --dir _masterV2 --models resnet50 vmamba_t efficientvmamba_s

Salida: <dir>/fig_erf_radial.{pdf,png} + tabla borde/centro por stdout.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

STYLE = {
    "resnet50": ("ResNet-50 (CNN)", "#1f77b4"),
    "resnet18": ("ResNet-18 (CNN)", "#17becf"),
    "efficientnet_b0": ("EfficientNet-B0 (CNN)", "#2ca02c"),
    "vmamba_t": ("VMamba-T (SSM jerárquico)", "#d62728"),
    "efficientvmamba_s": ("EffVMamba-S (SSM+conv)", "#ff7f0e"),
    "vim_s": ("Vim-S (SSM plano)", "#9467bd"),
}


def radial_profile(erf):
    """Perfil radial normalizado al centro. Devuelve (perfil, borde/centro)."""
    H, W = erf.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    prof = np.bincount(r.ravel(), weights=erf.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    prof = prof[:H // 2]
    prof = prof / prof[:3].mean()               # normaliza al centro
    edge = float(prof[-20:].mean())             # sensibilidad en el borde vs centro
    return prof, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_master", help="carpeta con los erf_*.npy")
    ap.add_argument("--models", nargs="+",
                    default=["resnet50", "vmamba_t", "efficientvmamba_s"])
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", args.dir))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    print(f"{'modelo':28s} {'borde/centro':>12s}  lectura")
    for m in args.models:
        path = os.path.join(root, f"erf_{m}.npy")
        if not os.path.exists(path):
            print(f"  (falta {path}, se omite)")
            continue
        prof, edge = radial_profile(np.load(path))
        lbl, c = STYLE.get(m, (m, None))
        tag = "GLOBAL (no decae)" if edge > 0.85 else (
            "local (decae)" if edge < 0.75 else "intermedio")
        print(f"{lbl:28s} {edge:12.2f}  {tag}")
        ax.plot(prof, color=c, lw=2.3, label=f"{lbl}  (borde/centro={edge:.2f})")

    ax.axhline(1.0, color="0.7", ls=":", lw=1)
    ax.set_xlabel("Distancia al centro (px)")
    ax.set_ylabel("|grad| del ERF relativo al centro")
    ax.set_title("Perfil radial del campo receptivo efectivo:\n"
                 "la CNN decae (local); Mamba se mantiene plano (global)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(root, "fig_erf_radial")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("Guardado:", out + ".png")


if __name__ == "__main__":
    main()

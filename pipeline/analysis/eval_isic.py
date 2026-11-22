"""Vuelca predicciones y probabilidades del test de ISIC por modelo (insumo para matrices de
confusión y curvas ROC del Cap 4). Requiere GPU y los checkpoints entrenados.

Carga el checkpoint entrenado, corre el split TEST congelado de ISIC y escribe un CSV por muestra:
  idx, y_true, prob_benign, prob_malignant, y_pred
Por defecto usa los pesos RAW (`model_raw`), que son los reportados en ISIC_RESULTS_V2.

Uso (por modelo):
  python pipeline/analysis/eval_isic.py --config pipeline/configs/resnet50.yaml \
      --isic-root <ruta> --split-file <ruta/isic_split.json> --ckpt runs/resnet50_isic/best.pth
  python pipeline/analysis/eval_isic.py --config pipeline/configs/vmamba_t.yaml ...

Salida: _master/isic_preds_<modelo>.csv
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from models.registry import build_model
from train import load_config, set_determinism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--isic-root", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pth del run ISIC del modelo")
    ap.add_argument("--weights", default="raw", choices=["raw", "ema"],
                    help="raw=model_raw (reportado), ema=model")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, "isic")
    set_determinism(cfg["seed"], cfg["deterministic"])
    cfg["isic_root"] = args.isic_root
    cfg["split_file"] = args.split_file
    cfg["data_root"] = None
    device = "cuda"

    test_ds = build_dataset("isic", "test", cfg)
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg["model"]["name"], cfg["num_classes"],
                        pretrained=False, pretrained_ckpt=None)
    ckpt = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
    key = "model_raw" if args.weights == "raw" else "model"
    state = ckpt.get(key, ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"[eval_isic] {cfg['model']['name']} | pesos={args.weights} ({key}) | test n={len(test_ds)}")

    name = cfg["model"]["name"]
    out_csv = args.out_csv or os.path.join("_master", f"isic_preds_{name}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    rows = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            probs = torch.softmax(logits.float(), dim=1).cpu()
            pred = probs.argmax(1)
            for i in range(x.size(0)):
                rows.append((int(y[i]), float(probs[i, 0]), float(probs[i, 1]), int(pred[i])))

    import csv
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "y_true", "prob_benign", "prob_malignant", "y_pred"])
        for i, r in enumerate(rows):
            w.writerow([i, *r])

    # sanidad: balanced acc rápida para confirmar orientación de clases
    import numpy as np
    yt = np.array([r[0] for r in rows]); yp = np.array([r[3] for r in rows])
    tp = ((yt == 1) & (yp == 1)).sum(); tn = ((yt == 0) & (yp == 0)).sum()
    p = (yt == 1).sum(); n = (yt == 0).sum()
    bacc = 0.5 * (tp / max(p, 1) + tn / max(n, 1))
    print(f"[eval_isic] guardado {out_csv} | balanced_acc(check)={bacc:.4f} "
          f"(debe coincidir con ISIC_RESULTS_V2)")


if __name__ == "__main__":
    main()

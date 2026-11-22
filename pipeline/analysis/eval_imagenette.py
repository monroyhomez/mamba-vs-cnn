"""Reevalúa el best.pth de Imagenette sobre la partición test' held-out (nunca vista durante
la selección de checkpoint) e imprime/vuelca el top-1/top-5 honesto. Requiere GPU y los
checkpoints entrenados.

El val_top1 histórico de entrenamiento (best.pth, monitor val_raw_top1 en trainer.py) se
calculó sobre el val COMPLETO — el mismo que se usó para decidir qué checkpoint guardar. Ese
número no sirve como resultado final (selección de modelo con el mismo set = circular). Este
script evalúa contra test' (mitad del val original, congelada por
pipeline/data/datasets.py::make_imagenette_split, jamás usada para elegir checkpoint).

Por defecto usa los pesos RAW (`model_raw`), que son los que monitorea/reporta el trainer
(ver trainer.py: el monitor es val_raw_top1, no la métrica EMA).

Uso (por modelo):
  python pipeline/analysis/eval_imagenette.py --config pipeline/configs/resnet50.yaml \
      --data-root ~/tesis_v3/data --split-file ~/tesis_v3/data/imagenette_split.json \
      --ckpt runs/resnet50_imagenette/best.pth

Salida: _master/imagenette_preds_<modelo>.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import build_imagenette_eval
from engine.metrics import classification_metrics
from models.registry import build_model
from train import load_config, set_determinism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pth del run Imagenette del modelo")
    ap.add_argument("--weights", default="raw", choices=["raw", "ema"],
                    help="raw=model_raw (el que monitorea/reporta el trainer), ema=model")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, "imagenette")
    set_determinism(cfg["seed"], cfg["deterministic"])
    device = "cuda"

    test_ds = build_imagenette_eval(args.data_root, args.split_file, "test", cfg["input_size"])
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg["model"]["name"], cfg["num_classes"],
                        pretrained=False, pretrained_ckpt=None)
    ckpt = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
    key = "model_raw" if args.weights == "raw" else "model"
    state = ckpt.get(key, ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"[eval_imagenette] {cfg['model']['name']} | pesos={args.weights} ({key}) | test'={len(test_ds)} "
          f"| ckpt.keys()={list(ckpt.keys())}")

    name = cfg["model"]["name"]
    out_csv = args.out_csv or os.path.join("_master", f"imagenette_preds_{name}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            y_true.append(y.numpy())
            y_prob.append(probs)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(1)

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "y_true", "y_pred", "pred_prob"])
        for i in range(len(y_true)):
            w.writerow([i, int(y_true[i]), int(y_pred[i]), float(y_prob[i, y_pred[i]])])

    # sanidad: recalcular top1/top5 con la misma función que usa el trainer, prefijo "test"
    m = classification_metrics(y_true, y_prob, cfg["num_classes"], prefix="test")
    print(f"[eval_imagenette] guardado {out_csv} | test_top1={m['test_top1']:.4f} "
          f"test_top5={m.get('test_top5', float('nan')):.4f} | n={len(y_true)}")


if __name__ == "__main__":
    main()

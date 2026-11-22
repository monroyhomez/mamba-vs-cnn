"""Punto de entrada único. Mismo script para los 7 modelos y los 2 datasets.
El modelo se elige por config; el dataset por --dataset. Todo lo demás es idéntico
por construcción (paridad).

Ejemplos:
  python pipeline/train.py --config pipeline/configs/resnet50.yaml \
      --dataset imagenette --data-root ~/tesis_v3/data
  python pipeline/train.py --config pipeline/configs/vmamba_t.yaml \
      --dataset isic --isic-root ~/tesis_v3/data/isic_dicm_17k \
      --split-file ~/tesis_v3/data/isic_split.json
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from engine.trainer import Trainer
from models.registry import build_model, count_parameters


def deep_update(base, override):
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path, dataset):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "configs", "_base.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    with open(config_path) as fh:
        cfg = deep_update(cfg, yaml.safe_load(fh))
    # aplicar overrides del dataset (monitor, num_classes, augment)
    ov = cfg.pop("dataset_overrides", {}).get(dataset, {})
    cfg = deep_update(cfg, ov)
    cfg["dataset"] = dataset
    return cfg


def set_determinism(seed, deterministic):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["imagenette", "isic"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--isic-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 época, validación rápida del pipeline")
    args = ap.parse_args()

    cfg = load_config(args.config, args.dataset)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.smoke:
        cfg["train"]["epochs"] = 1

    set_determinism(cfg["seed"], cfg["deterministic"])
    device = "cuda"

    # rutas que necesita build_dataset
    cfg["data_root"] = args.data_root
    cfg["isic_root"] = args.isic_root
    cfg["split_file"] = args.split_file

    train_ds = build_dataset(args.dataset, "train", cfg)
    val_ds = build_dataset(args.dataset, "val", cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg["model"]["name"], cfg["num_classes"],
                        pretrained=cfg["pretrained"],
                        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"))
    print(f"Modelo {cfg['model']['name']}: {count_parameters(model)/1e6:.2f}M parámetros")

    out_dir = args.out_dir or os.path.join("runs", f"{cfg['model']['name']}_{args.dataset}")
    trainer = Trainer(model, cfg, device, out_dir)
    result = trainer.fit(train_loader, val_loader)
    print("Resultado:", result, "| CSV:", trainer.csv_path)


if __name__ == "__main__":
    main()

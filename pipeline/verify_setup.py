"""Verificación de que los checkpoints ImageNet REALMENTE cargan en cada modelo Mamba.
NO entrena: solo construye y reporta params + las líneas `[ckpt] ... missing/unexpected`
que imprime registry._load_official_ckpt.

Criterio de éxito: missing ~0 (a lo sumo la cabeza). Si missing es de cientos, la config
NO coincide con el checkpoint y el modelo entrenaría desde cero (fallo silencioso).

Uso (desde comparativa_v3/):
    python pipeline/verify_setup.py --dataset imagenette
"""
import argparse
import os

import yaml

from models.registry import build_model, count_parameters

MAMBA = ["vmamba_t", "efficientvmamba_s", "vssd_micro", "vim_s"]


def _deep_update(base, ov):
    for k, v in (ov or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(cfg_path, dataset):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "configs", "_base.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    with open(cfg_path) as fh:
        cfg = _deep_update(cfg, yaml.safe_load(fh))
    _deep_update(cfg, cfg.pop("dataset_overrides", {}).get(dataset, {}))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    print("Modelo                | params  | resultado")
    print("-" * 60)
    for name in MAMBA:
        cfg_path = os.path.join(here, "configs", f"{name}.yaml")
        if not os.path.exists(cfg_path):
            print(f"{name:<22}| --      | SIN CONFIG")
            continue
        cfg = _load_config(cfg_path, args.dataset)
        ckpt = cfg["model"].get("pretrained_ckpt")
        print(f"\n### {name} | ckpt={ckpt}")
        if ckpt and not os.path.exists(os.path.expanduser(ckpt)):
            print(f"{name:<22}| --      | ⚠ CHECKPOINT NO EXISTE en disco")
            continue
        try:
            # build_model dispara el print '[ckpt] ... missing=N unexpected=M'
            model = build_model(cfg["model"]["name"], cfg["num_classes"],
                                pretrained=cfg["pretrained"], pretrained_ckpt=ckpt)
            print(f"{name:<22}| {count_parameters(model)/1e6:>5.2f}M | OK construido "
                  "(revisar arriba que missing≈0)")
        except NotImplementedError as e:
            print(f"{name:<22}| --      | PENDIENTE: {e}")
        except Exception as e:
            print(f"{name:<22}| --      | FALLO: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

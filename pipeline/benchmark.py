"""Entrypoint del benchmark de eficiencia (eje 2 de la tesis). Reproducible y con
PARIDAD por construcción: construye el modelo con el MISMO config y el MISMO builder
(registry) que el entrenamiento, y mide en bf16 (la precisión del entrenamiento).

Reemplaza el driver ad-hoc con el que se generaron los CSV previos (los `/tmp/bench_*.log`).

Ejemplos:
  # bf16 (primario, igual que el entrenamiento):
  python pipeline/benchmark.py --config pipeline/configs/vmamba_t.yaml

  # sensibilidad de precisión (fp32 además de bf16):
  python pipeline/benchmark.py --config pipeline/configs/resnet50.yaml --precision both

Notas:
- No carga datos: usa tensores aleatorios (el benchmark mide forward/backward, no accuracy).
- num_classes sale del config (imagenette=10, isic=2) igual que en train.py.
- FLOPs de Mamba: pasar el valor del paper por config (`benchmark.flops_paper_g_at_224`)
  para tener una columna independiente de fvcore (que sub-cuenta el selective-scan).
"""
import argparse
import os

import torch

from engine.benchmark import run_benchmark
from models.registry import build_model, count_parameters
from train import load_config, set_determinism


def _parse_resolutions(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="imagenette", choices=["imagenette", "isic"],
                    help="solo fija num_classes / tamaño de la cabeza (no se cargan datos)")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "both"],
                    help="bf16 = igual que el entrenamiento (default). fp32 = sensibilidad.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resolutions", default="224,384,512,768,1024")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-csv", default=None,
                    help="default: runs/_master/benchmark_<precision>.csv")
    args = ap.parse_args()

    cfg = load_config(args.config, args.dataset)
    set_determinism(cfg["seed"], cfg["deterministic"])
    device = "cuda"

    model = build_model(cfg["model"]["name"], cfg["num_classes"],
                        pretrained=cfg["pretrained"],
                        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"))
    name = cfg["model"]["name"]
    print(f"[benchmark] {name}: {count_parameters(model)/1e6:.2f}M parámetros | "
          f"precision={args.precision} | batch={args.batch_size}")

    flops_paper = cfg.get("benchmark", {}).get("flops_paper_g_at_224")
    resolutions = _parse_resolutions(args.resolutions)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [None],
              "both": [torch.bfloat16, None]}[args.precision]

    for amp_dtype in dtypes:
        prec = "bf16" if amp_dtype is not None else "fp32"
        out_csv = args.out_csv or os.path.join("runs", "_master", f"benchmark_{prec}.csv")
        print(f"[benchmark] === {prec} === → {out_csv}")
        run_benchmark(model, name, cfg["num_classes"], out_csv, device=device,
                      resolutions=resolutions, batch_size=args.batch_size,
                      amp_dtype=amp_dtype, repeats=args.repeats,
                      flops_paper_g_at_224=flops_paper)


if __name__ == "__main__":
    main()

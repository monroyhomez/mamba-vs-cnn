"""Figura de escalado de VRAM vs resolución (Cap 4, eje D / matiz jerarquía). Muestra que
el Mamba PLANO (Vim-S) OOMea a 768² mientras los jerárquicos (VMamba-T, EffVMamba-S) y las
CNN llegan a 1024². Mensaje: en SSM, la jerarquía no es opcional para alta resolución.

Datos:
  - 5 modelos: de _master/benchmark_bf16.csv (VRAM train reserved).
  - vim_s: del re-run post-parche multi-resolución (2026-07-03), hardcodeado aquí
    porque el CSV local aún tiene la corrida vieja (solo 224²). 768²/1024² = OOM.

Salida: _master/fig_scaling_vram.{pdf,png}
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH_CSV = os.path.join(ROOT, "_master", "benchmark_bf16.csv")
OUT_BASE = os.path.join(ROOT, "_master", "fig_scaling_vram")

RES = [224, 384, 512, 768, 1024]
GPU_GB = 31.5  # RTX 5000 Ada usable

LABEL = {"resnet18": "ResNet-18", "resnet50": "ResNet-50",
         "efficientnet_b0": "EfficientNet-B0", "vmamba_t": "VMamba-T (jerárquico)",
         "efficientvmamba_s": "EffVMamba-S (jerárquico)", "vim_s": "Vim-S (plano)"}
FAMILY = {"resnet18": "CNN", "resnet50": "CNN", "efficientnet_b0": "CNN",
          "vmamba_t": "Mamba", "efficientvmamba_s": "Mamba", "vim_s": "Mamba"}
STYLE = {"CNN": dict(color="#1f77b4"), "Mamba": dict(color="#d62728")}

# vim_s post-parche (reserved MB); None = OOM. Fuente: re-run 2026-07-03.
VIM_S = {224: 4194.7, 384: 11836.7, 512: 21129.3, 768: None, 1024: None}


def load_reserved():
    data = {}
    with open(BENCH_CSV) as fh:
        for row in csv.DictReader(fh):
            m, r = row["model"], int(row["resolution"])
            if m == "vim_s":
                continue  # se usa el hardcode nuevo
            v = row.get("vram_train_reserved_MB_mean", "")
            ok = row["status"] == "ok" and v not in ("", "nan")
            data.setdefault(m, {})[r] = float(v) / 1024 if ok else None
    data["vim_s"] = {r: (v / 1024 if v else None) for r, v in VIM_S.items()}
    return data


def main():
    data = load_reserved()
    fig, ax = plt.subplots(figsize=(8, 5.2))

    for m, series in data.items():
        fam = FAMILY[m]
        xs = [r for r in RES if series.get(r) is not None]
        ys = [series[r] for r in xs]
        emph = m in ("vim_s", "vmamba_t")
        ax.plot(xs, ys, marker="o", lw=2.4 if emph else 1.3,
                alpha=1.0 if emph else 0.55, zorder=3 if emph else 2,
                label=LABEL[m], **STYLE[fam],
                linestyle="-" if m != "vim_s" else "--")
        # marca OOM: X roja en la primera resolución que revienta
        oom_res = next((r for r in RES if series.get(r) is None), None)
        if oom_res is not None:
            ax.scatter([oom_res], [GPU_GB], marker="X", s=140, color=STYLE[fam]["color"],
                       edgecolors="black", linewidths=1.2, zorder=5)
            ax.annotate("OOM", (oom_res, GPU_GB), textcoords="offset points",
                        xytext=(6, -14), fontsize=8, color=STYLE[fam]["color"], weight="bold")

    ax.axhline(GPU_GB, color="0.4", ls=":", lw=1.2)
    ax.annotate("Límite GPU (32 GB Ada)", (224, GPU_GB), textcoords="offset points",
                xytext=(2, 4), fontsize=8, color="0.4")
    ax.set_xlabel("Resolución de entrada (px)")
    ax.set_ylabel("VRAM de entrenamiento (GB, reserved, batch=16, bf16)")
    ax.set_xticks(RES)
    ax.set_title("Escalado de memoria: el Mamba plano (Vim-S) OOMea a 768²;\n"
                 "los jerárquicos y las CNN llegan a 1024²", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", dpi=150, bbox_inches="tight")
    print("Guardado:", OUT_BASE + ".png")
    # resumen numérico
    for m in ("vim_s", "vmamba_t", "efficientvmamba_s"):
        maxres = max((r for r in RES if data[m].get(r) is not None), default=None)
        print(f"  {m}: max resolución sin OOM = {maxres}² "
              f"(VRAM@{maxres} = {data[m][maxres]:.1f} GB)")


if __name__ == "__main__":
    main()

"""Figura precisión-vs-eficiencia (Cap 4, eje A). Muestra que NINGÚN modelo domina:
Mamba está en la frontera de Pareto por *precisión*, la CNN por *velocidad/memoria*.

Datos:
  - Precisión: ISIC test held-out, balanced_acc, pesos RAW (protocolo corregido).
    Fuente: comparativa_v3/ISIC_RESULTS_V2.md (bloque B5 TEST-RAW).
  - Eficiencia: benchmark bf16 @224² (throughput y VRAM train reserved).
    Fuente: comparativa_v3/_master/benchmark_bf16.csv.

Salida: _master/fig_pareto.pdf y .png
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH_CSV = os.path.join(ROOT, "_master", "benchmark_bf16.csv")
OUT_BASE = os.path.join(ROOT, "_master", "fig_pareto")

# ISIC test held-out, balanced_acc (RAW) — de ISIC_RESULTS_V2.md §B5
ISIC_BACC = {
    "resnet18": 0.8183, "resnet50": 0.8317, "efficientnet_b0": 0.8451,
    "vmamba_t": 0.8531, "efficientvmamba_s": 0.8366, "vim_s": 0.8397,
}
FAMILY = {  # para color
    "resnet18": "CNN", "resnet50": "CNN", "efficientnet_b0": "CNN",
    "vmamba_t": "Mamba", "efficientvmamba_s": "Mamba", "vim_s": "Mamba",
}
LABEL = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
    "vmamba_t": "VMamba-T", "efficientvmamba_s": "EffVMamba-S", "vim_s": "Vim-S",
}
COLORS = {"CNN": "#1f77b4", "Mamba": "#d62728"}


def load_efficiency():
    """@224²: throughput_fps, vram_train_reserved_MB, params_M por modelo."""
    eff = {}
    with open(BENCH_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["resolution"] == "224" and row["status"] == "ok":
                eff[row["model"]] = {
                    "fps": float(row["throughput_fps_mean"]),
                    "vram": float(row["vram_train_reserved_MB_mean"]) / 1024.0,  # GB
                    "params": float(row["params_M"]),
                }
    return eff


def pareto_front(points, x_better="high", y_better="high"):
    """Devuelve los nombres NO dominados. points: {name:(x,y)}."""
    names = list(points)
    front = []
    for a in names:
        xa, ya = points[a]
        dominated = False
        for b in names:
            if b == a:
                continue
            xb, yb = points[b]
            xdom = xb >= xa if x_better == "high" else xb <= xa
            ydom = yb >= ya if y_better == "high" else yb <= ya
            xstr = xb > xa if x_better == "high" else xb < xa
            ystr = yb > ya if y_better == "high" else yb < ya
            if xdom and ydom and (xstr or ystr):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def _panel(ax, points, xlabel, x_better, title):
    front = set(pareto_front(points, x_better=x_better, y_better="high"))
    # línea de frontera
    fr = sorted(front, key=lambda m: points[m][0])
    ax.plot([points[m][0] for m in fr], [points[m][1] for m in fr],
            "--", color="0.6", lw=1, zorder=1)
    for m, (x, y) in points.items():
        fam = FAMILY[m]
        ax.scatter(x, y, s=40 + 12 * ISIC_PARAMS[m], c=COLORS[fam],
                   edgecolors="black" if m in front else "none",
                   linewidths=1.4, alpha=0.9, zorder=3)
        dy = 0.0012 if m not in ("resnet50",) else -0.0028
        ax.annotate(LABEL[m], (x, y), textcoords="offset points",
                    xytext=(7, 5), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ISIC balanced accuracy (test)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    eff = load_efficiency()
    global ISIC_PARAMS
    ISIC_PARAMS = {m: eff[m]["params"] for m in eff}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    pts_fps = {m: (eff[m]["fps"], ISIC_BACC[m]) for m in eff}
    _panel(axes[0], pts_fps, "Throughput @224² (img/s, ↑ mejor)", "high",
           "Precisión vs. velocidad")
    axes[0].set_xscale("log")

    pts_vram = {m: (eff[m]["vram"], ISIC_BACC[m]) for m in eff}
    _panel(axes[1], pts_vram, "VRAM entrenamiento @224² (GB, ↓ mejor)", "low",
           "Precisión vs. memoria")

    # leyenda de familia + nota
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[f],
                      markersize=9, label=f) for f in ("CNN", "Mamba")]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                          markeredgecolor="black", markersize=9, label="frontera de Pareto"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Nadie domina: Mamba lidera precisión; la CNN, velocidad y memoria "
                 "(tamaño ∝ parámetros)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", dpi=150, bbox_inches="tight")
    print("Frontera Pareto (precisión vs fps):",
          pareto_front(pts_fps, x_better="high", y_better="high"))
    print("Frontera Pareto (precisión vs VRAM):",
          pareto_front(pts_vram, x_better="low", y_better="high"))
    print("Guardado:", OUT_BASE + ".pdf/.png")


if __name__ == "__main__":
    main()

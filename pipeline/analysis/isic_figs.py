"""Figuras y métricas de ISIC (Cap 4.2), desde una fuente única: los CSV exportados por
eval_isic.py (predicciones por muestra + curvas por época). Corre LOCAL.

Genera (en _master/):
  - fig_isic_roc.png/.pdf         curvas ROC de los 6 modelos + AUROC
  - fig_isic_confusion.png/.pdf   matrices de confusión (2x3)
  - fig_isic_curves.png/.pdf      balanced-acc (RAW) por época + train loss
  - imprime la tabla de métricas (misma fuente que las figuras → consistente con tab:isic)

Uso: python pipeline/analysis/isic_figs.py --dir redaccion_20260704
"""
import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

MODELS = [  # orden y etiqueta; familia para color
    ("resnet18", "ResNet-18", "CNN"),
    ("resnet50", "ResNet-50", "CNN"),
    ("efficientnet_b0", "EfficientNet-B0", "CNN"),
    ("efficientvmamba_s", "EffVMamba-S", "Mamba"),
    ("vim_s", "Vim-S", "Mamba"),
    ("vmamba_t", "VMamba-T", "Mamba"),
]
COL = {"CNN": "#1f77b4", "Mamba": "#d62728"}
CMAP_LINE = plt.cm.tab10


def load_preds(d, m):
    yt, pm, yp = [], [], []
    with open(os.path.join(d, f"isic_preds_{m}.csv")) as fh:
        for r in csv.DictReader(fh):
            yt.append(int(r["y_true"])); pm.append(float(r["prob_malignant"])); yp.append(int(r["y_pred"]))
    return np.array(yt), np.array(pm), np.array(yp)


def metrics(yt, pm, yp):
    tp = int(((yt == 1) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    rec = tp / (tp + fn); spec = tn / (tn + fp); bacc = 0.5 * (rec + spec)
    prec = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(cm=[[tn, fp], [fn, tp]], bacc=bacc, mcc=mcc, rec=rec, spec=spec, f1=f1)


def roc_curve(yt, score):
    order = np.argsort(-score)
    yt = yt[order]
    P = (yt == 1).sum(); N = (yt == 0).sum()
    tpr = np.cumsum(yt == 1) / P
    fpr = np.cumsum(yt == 0) / N
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="redaccion_20260704")
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ddir = os.path.join(root, args.dir)
    out = os.path.join(root, "_master")

    # ---- métricas + ROC ----
    fig_r, ax_r = plt.subplots(figsize=(6, 5.6))
    print(f"{'modelo':16s} {'bacc':>6} {'mcc':>6} {'auroc':>6} {'recall':>6} {'spec':>6} {'f1':>6}")
    rows_cm = []
    for i, (m, lbl, fam) in enumerate(MODELS):
        yt, pm, yp = load_preds(ddir, m)
        mt = metrics(yt, pm, yp)
        fpr, tpr, auc = roc_curve(yt, pm)
        print(f"{lbl:16s} {mt['bacc']:.4f} {mt['mcc']:.4f} {auc:.4f} "
              f"{mt['rec']:.4f} {mt['spec']:.4f} {mt['f1']:.4f}")
        ax_r.plot(fpr, tpr, lw=2, color=CMAP_LINE(i), label=f"{lbl} (AUC={auc:.3f})")
        rows_cm.append((lbl, mt["cm"], mt["bacc"]))
    ax_r.plot([0, 1], [0, 1], "k:", lw=1)
    ax_r.set_xlabel("Tasa de falsos positivos"); ax_r.set_ylabel("Tasa de verdaderos positivos")
    ax_r.set_title("Curvas ROC en ISIC (test held-out, clase positiva = maligno)")
    ax_r.legend(fontsize=8, loc="lower right"); ax_r.grid(alpha=0.3)
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(out, "fig_isic_roc.pdf"), bbox_inches="tight")
    fig_r.savefig(os.path.join(out, "fig_isic_roc.png"), dpi=150, bbox_inches="tight")

    # ---- matrices de confusión (2x3) ----
    fig_c, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (lbl, cm, bacc) in zip(axes.ravel(), rows_cm):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r,c]}", ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benigno", "Maligno"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Benigno", "Maligno"], fontsize=8)
        ax.set_xlabel("Predicho", fontsize=8); ax.set_ylabel("Real", fontsize=8)
        ax.set_title(f"{lbl}  (bal-acc {bacc:.3f})", fontsize=9)
    fig_c.suptitle("Matrices de confusión en ISIC (test, n=2240)", fontsize=12)
    fig_c.tight_layout(rect=[0, 0, 1, 0.97])
    fig_c.savefig(os.path.join(out, "fig_isic_confusion.pdf"), bbox_inches="tight")
    fig_c.savefig(os.path.join(out, "fig_isic_confusion.png"), dpi=150, bbox_inches="tight")

    # ---- curvas por época (RAW) ----
    fig_t, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (m, lbl, fam) in enumerate(MODELS):
        ep, tl, ba = [], [], []
        with open(os.path.join(ddir, f"isic_curve_{m}.csv")) as fh:
            for r in csv.DictReader(fh):
                ep.append(int(r["epoch"])); tl.append(float(r["train_loss"]))
                ba.append(float(r["val_raw_balanced_acc"]))
        a1.plot(ep, tl, color=CMAP_LINE(i), lw=1.6, label=lbl)
        a2.plot(ep, ba, color=CMAP_LINE(i), lw=1.6, label=lbl)
    a1.set_xlabel("Época"); a1.set_ylabel("Train loss"); a1.set_title("Pérdida de entrenamiento")
    a2.set_xlabel("Época"); a2.set_ylabel("Val balanced-acc (RAW)"); a2.set_title("Balanced accuracy de validación")
    a2.legend(fontsize=8, loc="lower right"); a1.grid(alpha=0.3); a2.grid(alpha=0.3)
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(out, "fig_isic_curves.pdf"), bbox_inches="tight")
    fig_t.savefig(os.path.join(out, "fig_isic_curves.png"), dpi=150, bbox_inches="tight")
    print("Guardado: fig_isic_roc, fig_isic_confusion, fig_isic_curves (en _master/)")


if __name__ == "__main__":
    main()

"""Métricas de clasificación. Devuelve un dict con prefijo de split.
Para ISIC (binario) incluye MCC, balanced accuracy, recall melanoma, specificity,
F1 y AUROC — todas reportadas. Para Imagenette incluye top-1 y top-5."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, f1_score, roc_auc_score, confusion_matrix, top_k_accuracy_score,
)


def classification_metrics(y_true, y_prob, num_classes, prefix="val"):
    """y_true: (N,) int. y_prob: (N, C) softmax. Devuelve dict de floats."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    m = {
        f"{prefix}_top1": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # Top-5 solo tiene sentido con >5 clases (Imagenette). Validar que NO sea 0 (bug previo).
    if num_classes > 5:
        m[f"{prefix}_top5"] = float(
            top_k_accuracy_score(y_true, y_prob, k=5, labels=list(range(num_classes))))
    if num_classes == 2:
        # clase positiva = melanoma = 1
        m[f"{prefix}_recall_mel"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m[f"{prefix}_specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
        try:
            m[f"{prefix}_auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            m[f"{prefix}_auroc"] = float("nan")
    return m

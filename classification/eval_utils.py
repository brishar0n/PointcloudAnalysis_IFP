"""
eval_utils.py — Shared evaluation helpers.

Used by dl_classifier.py, dl_loco_evaluation.py, dl_train_final_model.py.

Functions:
    print_metrics()        - Print accuracy + sidewalk F1 for a split
    plot_confusion()       - Normalised confusion matrix
    plot_confidence()      - Confidence histogram per class
    full_evaluation()      - Run all three for a given split
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

CLASS_NAMES  = ["other", "sidewalk", "street"]
CLASS_LABELS = [0, 1, 2]


# ── Metrics ───────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, split_name=""):
    """
    Print balanced accuracy, sidewalk F1, and per-class metrics.
    Shows only per-class rows (other/sidewalk/street) —
    accuracy, macro avg and weighted avg rows are excluded.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sw_f1   = f1_score(y_true, y_pred, labels=[1],
                       average=None, zero_division=0)[0]

    print(f"\n{'─'*50}")
    print(f"  {split_name} Results")
    print(f"{'─'*50}")
    print(f"  Balanced Accuracy : {bal_acc*100:.1f}%")
    print(f"  Sidewalk F1       : {sw_f1:.3f}")

    # Print only per-class rows — skip accuracy/macro/weighted avg
    report = classification_report(
        y_true, y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
    print(f"  {'─'*52}")
    for cls in CLASS_NAMES:
        r = report[cls]
        print(f"  {cls:<12} {r['precision']:>10.2f} {r['recall']:>8.2f} "
              f"{r['f1-score']:>8.2f} {int(r['support']):>10,}")

    return bal_acc, sw_f1


# ── Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, title, save_path):
    """Plot and save normalised confusion matrix."""
    cm      = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


# ── Confidence Histogram ──────────────────────────────────────────────────

def plot_confidence(model, scaler, X, device, title, save_path,
                    batch_size=512):
    """
    Plot confidence histogram — for each class, show distribution
    of prediction probabilities.

    X-axis: confidence (0 to 1)
    Y-axis: number of predictions at that confidence level
    One histogram per class (other / sidewalk / street)

    High bars near 1.0 = model is confident.
    Spread out bars = model is uncertain.
    """
    model.eval()
    X_sc     = scaler.transform(X).astype("float32")
    X_tensor = torch.FloatTensor(X_sc).to(device)

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch  = X_tensor[i:i+batch_size]
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)   # (N, 3)
    pred_cls  = all_probs.argmax(axis=1)

    colors = ["#1565C0", "#2E7D32", "#C62828"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = pred_cls == i
        if mask.sum() == 0:
            axes[i].set_title(f"{cls_name}\n(no predictions)")
            continue
        confidences = all_probs[mask, i]
        axes[i].hist(confidences, bins=20, color=color,
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(
            f"{cls_name}\n"
            f"n={mask.sum():,} | "
            f"mean conf={confidences.mean():.2f}"
        )
        axes[i].set_xlabel("Confidence (probability)")
        axes[i].set_ylabel("Number of segments")
        axes[i].set_xlim(0, 1)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confidence histogram saved -> {save_path}")


# ── Full evaluation ───────────────────────────────────────────────────────

def full_evaluation(model, scaler, device,
                    X_train, y_train,
                    X_val,   y_val,
                    X_test,  y_test,
                    predict_fn,
                    prefix,
                    results_dir="results",
                    show_test=True):
    """
    Run full evaluation: train + val metrics always shown.
    Test metrics shown only when show_test=True (e.g. LOCO and single city).
    For final model training, show_test=False — LOCO is the real test.

    predict_fn : callable that takes (model, scaler, X, device) -> y_pred
    prefix     : used for filenames e.g. "riga_mlp"
    show_test  : if False, skips test evaluation (use for dl_train_final_model)
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Train metrics ──────────────────────────────────────────────────────
    print("\nEvaluating on training set...")
    y_train_pred                = predict_fn(model, scaler, X_train, device)
    train_acc, train_sw_f1      = print_metrics(
        y_train, y_train_pred, split_name="TRAIN")

    # ── Validation metrics ─────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    y_val_pred               = predict_fn(model, scaler, X_val, device)
    val_acc, val_sw_f1       = print_metrics(
        y_val, y_val_pred, split_name="VALIDATION")

    # ── Test metrics (optional) ────────────────────────────────────────────
    test_acc    = None
    test_sw_f1  = None
    y_test_pred = None

    if show_test:
        print("\nEvaluating on test set...")
        y_test_pred            = predict_fn(model, scaler, X_test, device)
        test_acc, test_sw_f1   = print_metrics(
            y_test, y_test_pred, split_name="TEST")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION SUMMARY — {prefix.upper()}")
    print(f"{'='*55}")
    print(f"  {'Split':<12} {'Balanced Acc':>14} {'Sidewalk F1':>12}")
    print(f"  {'─'*40}")
    print(f"  {'Train':<12} {train_acc*100:>13.1f}% {train_sw_f1:>12.3f}")
    print(f"  {'Validation':<12} {val_acc*100:>13.1f}% {val_sw_f1:>12.3f}")
    if show_test and test_acc is not None:
        print(f"  {'Test':<12} {test_acc*100:>13.1f}% {test_sw_f1:>12.3f}")
    else:
        print(f"  {'Test':<12} {'See LOCO results':>26}")
    print(f"{'='*55}")

    # ── Confusion matrix (test if available, else val) ─────────────────────
    cm_data  = y_test_pred if show_test and y_test_pred is not None else y_val_pred
    cm_true  = y_test      if show_test and y_test_pred is not None else y_val
    cm_label = "Test Set"  if show_test and y_test_pred is not None else "Validation Set"
    plot_confusion(
        cm_true, cm_data,
        title=f"Confusion Matrix — {prefix.upper()} — {cm_label}",
        save_path=f"{results_dir}/{prefix}_confusion.png"
    )

    # ── Confidence histogram (val set) ─────────────────────────────────────
    plot_confidence(
        model, scaler, X_val, device,
        title=f"Prediction Confidence — {prefix.upper()} — Validation Set",
        save_path=f"{results_dir}/{prefix}_confidence.png"
    )

    return {
        "train_acc"   : train_acc,
        "train_sw_f1" : train_sw_f1,
        "val_acc"     : val_acc,
        "val_sw_f1"   : val_sw_f1,
        "test_acc"    : test_acc,
        "test_sw_f1"  : test_sw_f1,
    }

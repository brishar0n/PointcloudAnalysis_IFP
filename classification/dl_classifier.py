"""
dl_classifier.py — Deep Learning classifier for sidewalk point clouds.

Uses a Multi-Layer Perceptron (MLP) neural network on segment features.
Same segment pipeline as segment_classifier.py but replaces Random Forest
with a neural network.

Why MLP?
- Neural network = deep learning
- Uses same precomputed segment features — no raw point processing needed
- Faster to train than PointNet/RandLA-Net
- Can learn more complex non-linear patterns than Random Forest
- Better cross-city generalization through dropout regularization

Architecture:
    Input (123 features)
    -> Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    -> Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    -> Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
    -> Dense(3)   + Softmax
    Output (3 classes: other, sidewalk, street)

Usage:
    python dl_classifier.py --city riga
    python dl_classifier.py --city riga --epochs 100
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             balanced_accuracy_score,
                             f1_score)

from utils import (
    SEED, SAMPLE_SIZE,
    load_city, sample_and_filter, build_segments,
    add_context_features, evaluate
)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── MLP Architecture ──────────────────────────────────────────────────────

class SidewalkMLP(nn.Module):
    """
    Multi-Layer Perceptron for sidewalk classification.
    """

    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super(SidewalkMLP, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ── Training function ─────────────────────────────────────────────────────

def train_mlp(X_train, y_train, X_val, y_val,
              epochs=50, batch_size=512, lr=0.001):
    """
    Train MLP with class weighting to handle imbalance.

    WHY CLASS WEIGHTS?
    Same reason as RF — sidewalk is minority class.
    We compute inverse frequency weights so model pays
    more attention to sidewalk during training.

    WHY DROPOUT?
    Prevents the model from memorising training city patterns.
    Forces it to learn generalizable features — better cross-city performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Scale features
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)

    # Compute class weights for imbalance
    # Always produce weights for all 3 classes even if some are missing
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 3:
        print(f"  [WARNING] Only {len(classes)} class(es) in training data: {list(classes)}")
        print(f"  Results may be unreliable — this city may not have enough labels.")
    weight_vec = np.ones(3, dtype=np.float32)
    raw_w      = 1.0 / counts
    raw_w      = raw_w / raw_w.sum() * len(classes)
    for c, w in zip(classes, raw_w):
        weight_vec[c] = w
    class_weights   = torch.FloatTensor(weight_vec).to(device)
    weights_display = {i: round(float(weight_vec[i]), 3) for i in range(3)}
    print(f"  Class weights: {weights_display}")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train_sc).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val_sc).to(device)
    y_vl = torch.LongTensor(y_val).to(device)

    # DataLoader
    dataset    = TensorDataset(X_tr, y_tr)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model     = SidewalkMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    train_losses = []
    val_accs     = []
    best_val_acc = 0
    best_model   = None

    print(f"\n  Training MLP for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_vl)
            val_preds   = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc     = balanced_accuracy_score(y_val, val_preds)

        train_losses.append(total_loss / len(loader))
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_losses[-1]:.4f} | "
                  f"Val Balanced Acc: {val_acc*100:.1f}%")

    # Load best model
    model.load_state_dict(best_model)
    print(f"\n  Best validation balanced accuracy: {best_val_acc*100:.1f}%")

    return model, scaler, device, train_losses, val_accs


def predict_mlp(model, scaler, X, device, batch_size=512):
    """Run inference on feature matrix X."""
    model.eval()
    X_sc      = scaler.transform(X).astype(np.float32)
    X_tensor  = torch.FloatTensor(X_sc).to(device)
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch   = X_tensor[i:i+batch_size]
            outputs = model(batch)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# ── Standalone run ────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city",   default="riga")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Deep Learning Classifier (MLP) — {args.city.upper()}")
    print(f"{'='*60}")

    # ── Step 1: Load and process ──────────────────────────────────────────
    xyz, labels, features, feat_names             = load_city(args.city)
    xyz_s, labels_s, features_s                   = sample_and_filter(
                                                        xyz, labels, features,
                                                        feat_names, args.sample)
    seg_xyz, seg_feats, seg_labels, \
        unique_voxels, inverse_idx                 = build_segments(
                                                        xyz_s, labels_s, features_s)
    X_seg, nbr_indices                             = add_context_features(
                                                        seg_xyz, seg_feats)

    # ── Step 2: Spatial tile split ────────────────────────────────────────
    N_TILES      = 4
    x_min, x_max = seg_xyz[:, 0].min(), seg_xyz[:, 0].max()
    y_min, y_max = seg_xyz[:, 1].min(), seg_xyz[:, 1].max()
    tile_x       = np.floor((seg_xyz[:, 0] - x_min) / (x_max - x_min) * N_TILES
                            ).astype(int).clip(0, N_TILES - 1)
    tile_y       = np.floor((seg_xyz[:, 1] - y_min) / (y_max - y_min) * N_TILES
                            ).astype(int).clip(0, N_TILES - 1)
    tile_id      = tile_x * N_TILES + tile_y

    # Pick test tile with all 3 classes
    test_tile = np.bincount(tile_id).argmax()
    classes_in_test = np.unique(seg_labels[tile_id == test_tile])
    if len(classes_in_test) < 3:
        for t in range(N_TILES * N_TILES):
            mask_t = tile_id == t
            if mask_t.sum() < 50:
                continue
            if len(np.unique(seg_labels[mask_t])) >= 3:
                test_tile = t
                break

    test_mask  = tile_id == test_tile
    remain_mask = tile_id != test_tile

    X_remain, y_remain = X_seg[remain_mask], seg_labels[remain_mask]
    X_test,   y_test   = X_seg[test_mask],   seg_labels[test_mask]

    # ── 60/20/20 split ────────────────────────────────────────────────────
    # Test  = held-out spatial tile (~20% of data)
    # Val   = 25% of remaining data -> gives ~20% of total
    # Train = 75% of remaining data -> gives ~60% of total
    val_size = int(len(X_remain) * 0.25)
    X_val    = X_remain[:val_size]
    y_val    = y_remain[:val_size]
    X_train  = X_remain[val_size:]
    y_train  = y_remain[val_size:]

    total = len(X_seg)
    print(f"\n60/20/20 Spatial Split — test tile: {test_tile}")
    print(f"  Train: {len(X_train):,} ({100*len(X_train)/total:.0f}%)"
          f" | Val: {len(X_val):,} ({100*len(X_val)/total:.0f}%)"
          f" | Test: {len(X_test):,} ({100*len(X_test)/total:.0f}%)")

    # ── Step 3: Train MLP ─────────────────────────────────────────────────
    model, scaler, device, train_losses, val_accs = train_mlp(
        X_train, y_train, X_val, y_val, epochs=args.epochs)

    # ── Step 4: Full evaluation — train / val / test ───────────────────────
    from eval_utils import full_evaluation
    full_evaluation(
        model, scaler, device,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        predict_fn  = predict_mlp,
        prefix      = f"{args.city}_mlp",
        results_dir = "results"
    )

    # ── Step 5: Training curves ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, color="#1565C0")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot([v*100 for v in val_accs], color="#2E7D32")
    axes[1].set_title("Validation Balanced Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy (%)")

    plt.suptitle(f"MLP Training — {args.city.upper()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/{args.city}_mlp_training.png", dpi=150)
    plt.close()

    # ── Step 6: Save model ────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(),
               f"models/{args.city}_mlp_classifier.pt")
    import joblib
    joblib.dump(scaler, f"models/{args.city}_mlp_scaler.joblib")
    print(f"\nMLP model saved -> models/{args.city}_mlp_classifier.pt")

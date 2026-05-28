"""
dl_classifier.py — Deep Learning classifiers for sidewalk point clouds.

Two-stage pipeline:
  Stage 2 only — CSF handles ground/non-ground separation (Stage 1).
  This file trains a binary MLP: sidewalk vs street on ground-only segments.

Why binary instead of 3-class?
  The original 3-class MLP struggled because sidewalk and street are both
  flat ground surfaces — subtle differences in intensity/height get drowned
  out when the model also has to learn building/tree separation.
  CSF handles non-ground removal geometrically (it was designed for this).
  The binary MLP can then focus entirely on the hard problem.

Architecture (SidewalkStreetMLP):
    Input (n_features)
    -> Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    -> Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    -> Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
    -> Dense(2)   + Softmax
    Output: 0 = street, 1 = sidewalk

Legacy 3-class SidewalkMLP kept for backward compatibility.
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             classification_report)

from utils import (
    SEED, SAMPLE_SIZE,
    load_city, sample_and_filter, build_segments,
    add_context_features, evaluate,
    MODEL_SIDEWALK, MODEL_STREET
)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Binary MLP: sidewalk vs street ────────────────────────────────────────

class SidewalkStreetMLP(nn.Module):
    """
    Binary MLP — sidewalk(1) vs street(0).
    Only ever sees ground segments (CSF already removed non-ground).
    Smaller than the old 3-class model: simpler problem, less capacity needed.
    """
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


def train_mlp_binary(X_train, y_train, X_val, y_val,
                     epochs=50, batch_size=512, lr=0.001):
    """
    Train binary MLP: street(0) vs sidewalk(1).

    y_train / y_val must already be binary:
        MODEL_STREET   (2 in 3-class) -> 0
        MODEL_SIDEWALK (1 in 3-class) -> 1

    Returns: model, scaler, device, train_losses, val_accs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_train).astype(np.float32)
    X_vl_sc    = scaler.transform(X_val).astype(np.float32)

    # Class weights — sidewalk is minority vs street
    classes, counts = np.unique(y_train, return_counts=True)
    weight_vec = np.ones(2, dtype=np.float32)
    raw_w      = 1.0 / counts
    raw_w      = raw_w / raw_w.sum() * len(classes)
    for c, w in zip(classes, raw_w):
        weight_vec[c] = w
    class_weights   = torch.FloatTensor(weight_vec).to(device)
    weights_display = {i: round(float(weight_vec[i]), 3) for i in range(2)}
    print(f"  Class weights (street=0, sidewalk=1): {weights_display}")

    X_tr = torch.FloatTensor(X_tr_sc).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_vl_sc).to(device)
    y_vl = torch.LongTensor(y_val).to(device)

    dataset   = TensorDataset(X_tr, y_tr)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = SidewalkStreetMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_accs     = []
    best_val_acc = 0.0
    best_state   = None

    print(f"\n  Training binary MLP (sidewalk vs street) for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl).argmax(dim=1).cpu().numpy()
            val_acc   = balanced_accuracy_score(y_val, val_preds)

        train_losses.append(total_loss / len(loader))
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            sw_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_losses[-1]:.4f} | "
                  f"Val Balanced Acc: {val_acc*100:.1f}% | "
                  f"Sidewalk F1: {sw_f1:.3f}")

    model.load_state_dict(best_state)
    print(f"\n  Best val balanced accuracy: {best_val_acc*100:.1f}%")
    return model, scaler, device, train_losses, val_accs


def predict_mlp_binary(model, scaler, X, device, batch_size=512):
    """
    Run binary inference. Returns 0 (street) or 1 (sidewalk).
    """
    model.eval()
    X_sc     = scaler.transform(X).astype(np.float32)
    X_tensor = torch.FloatTensor(X_sc).to(device)
    preds    = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            preds.extend(
                model(X_tensor[i:i+batch_size]).argmax(dim=1).cpu().numpy()
            )
    return np.array(preds)


def predict_proba_binary(model, scaler, X, device, batch_size=512):
    """
    Return softmax probabilities for binary model.
    Shape: (n_segments, 2) — columns are [street_prob, sidewalk_prob].
    Used by pseudo-label fine-tuning to filter by confidence.
    """
    model.eval()
    X_sc     = scaler.transform(X).astype(np.float32)
    X_tensor = torch.FloatTensor(X_sc).to(device)
    probs    = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            logits = model(X_tensor[i:i+batch_size])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


# ── Legacy 3-class MLP — kept for backward compatibility ─────────────────

class SidewalkMLP(nn.Module):
    """
    Legacy 3-class MLP. Kept so old saved models can still be loaded.
    New training uses SidewalkStreetMLP (binary) with CSF for non-ground.
    """
    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_mlp(X_train, y_train, X_val, y_val,
              epochs=50, batch_size=512, lr=0.001):
    """Legacy 3-class trainer. Use train_mlp_binary() for new pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_train).astype(np.float32)
    X_vl_sc    = scaler.transform(X_val).astype(np.float32)
    classes, counts = np.unique(y_train, return_counts=True)
    weight_vec = np.ones(3, dtype=np.float32)
    raw_w      = 1.0 / counts
    raw_w      = raw_w / raw_w.sum() * len(classes)
    for c, w in zip(classes, raw_w):
        weight_vec[c] = w
    class_weights = torch.FloatTensor(weight_vec).to(device)
    X_tr  = torch.FloatTensor(X_tr_sc).to(device)
    y_tr  = torch.LongTensor(y_train).to(device)
    X_vl  = torch.FloatTensor(X_vl_sc).to(device)
    y_vl  = torch.LongTensor(y_val).to(device)
    loader    = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=batch_size, shuffle=True)
    model     = SidewalkMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_losses, val_accs = [], []
    best_val_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl).argmax(dim=1).cpu().numpy()
            val_acc   = balanced_accuracy_score(y_val, val_preds)
        train_losses.append(total_loss / len(loader))
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_losses[-1]:.4f} | "
                  f"Val Balanced Acc: {val_acc*100:.1f}%")
    model.load_state_dict(best_state)
    return model, scaler, device, train_losses, val_accs


def predict_mlp(model, scaler, X, device, batch_size=512):
    """Legacy 3-class predict. Use predict_mlp_binary() for new pipeline."""
    model.eval()
    X_sc     = scaler.transform(X).astype(np.float32)
    X_tensor = torch.FloatTensor(X_sc).to(device)
    preds    = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            preds.extend(
                model(X_tensor[i:i+batch_size]).argmax(dim=1).cpu().numpy()
            )
    return np.array(preds)

"""MLP on Split MNIST with IntervalActivation + gradient cosine tracking.

Simpler model/dataset to verify GradientCosineTracker for drift vs align.
Run: python experiments/simple_mlp/split_mnist_mlp.py [--real_bounds]

Without --real_bounds: artificially splits dims so both losses produce non-zero
gradients (half tight, half shifted). With --real_bounds: uses cumulative
bounds as-is (align loss may be 0 if no overlap violations).
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from copy import deepcopy
import os
import csv

from utils.gradient_cosine import GradientCosineTracker
from models.layers.interval_activation import IntervalActivation
from regularization.interval_regularization import IntervalPenalization


class MLPFeatureExtractor(nn.Module):
    """Matches ViT interface. When `prompt` is an MLPFeatureExtractor instance
    it uses that module's params (old snapshot for drift loss)."""

    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, prompt=None, q=None, train=False, task_id=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if isinstance(prompt, MLPFeatureExtractor):
            with torch.no_grad():
                out = prompt.net(x)
        else:
            out = self.net(x)
        return out.unsqueeze(1), None


class MLPModel(nn.Module):
    """feature_extractor + classifier head with IntervalActivation."""

    def __init__(self, input_dim=784, hidden_dim=128, num_classes=5):
        super().__init__()
        self.feature_extractor = MLPFeatureExtractor(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            IntervalActivation((hidden_dim,), use_non_linear_transform=False),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, train=False, prompt=None):
        feat, _ = self.feature_extractor(x)
        feat = feat[:, 0, :]
        logits = self.classifier(feat)
        return logits, None


def get_split_mnist(task_id, batch_size=256):
    """Split MNIST: task 0 = digits 0-4, task 1 = digits 5-9."""
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST('~/data/mnist', train=True, download=True, transform=t)
    ds_test = datasets.MNIST('~/data/mnist', train=False, download=True, transform=t)

    lo, hi = (0, 5) if task_id == 0 else (5, 10)

    def filter_ds(ds):
        idx = (ds.targets >= lo) & (ds.targets < hi)
        data = (ds.data[idx].float() / 255.0 - 0.1307) / 0.3081
        data = data.unsqueeze(1)
        targets = ds.targets[idx] - lo
        return data, targets

    x_tr, y_tr = filter_ds(ds_train)
    x_te, y_te = filter_ds(ds_test)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb, train=False)
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
    return correct / total


def fill_ia_buffers(model, loader, device, num_batches=50):
    """Run eval pass to fill IntervalActivation.test_act_buffer."""
    model.eval()
    for i, (xb, yb) in enumerate(loader):
        if i >= num_batches:
            break
        xb = xb.to(device)
        with torch.no_grad():
            model(xb, train=False)


def set_artificial_bounds(model):
    """Split dims: half shrunk (drift), half shifted far away (align).
    Ensures both losses produce non-zero gradients for cosine measurement."""
    for layer in model.classifier:
        if isinstance(layer, IntervalActivation) and layer.min is not None:
            D = layer.min.shape[0]
            half = D // 2
            center = (layer.min[:half] + layer.max[:half]) / 2
            radius = (layer.max[:half] - layer.min[:half]) / 2 * 0.3
            layer.min[:half] = center - radius
            layer.max[:half] = center + radius
            layer.min[half:] = 50.0
            layer.max[half:] = 100.0


def print_cosine_summary(csv_path):
    cosines = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row['interval_drift__align']
            try:
                cosines.append(float(val))
            except (ValueError, TypeError):
                pass
    cosines = np.array(cosines)
    valid = cosines[~np.isnan(cosines)]
    print(f"Steps recorded: {len(cosines)}")
    print(f"  NaN count: {np.isnan(cosines).sum()}")
    print(f"  Valid steps: {len(valid)}")
    if len(valid) > 0:
        print(f"  Cosine mean: {valid.mean():.4f}")
        print(f"  Cosine std:  {valid.std():.4f}")
        print(f"  Cosine min:  {valid.min():.4f}")
        print(f"  Cosine max:  {valid.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="MLP Split-MNIST gradient cosine test")
    parser.add_argument('--real_bounds', action='store_true',
                        help='Use cumulative bounds as-is (align=0 when no violations)')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs0', type=int, default=3)
    parser.add_argument('--epochs1', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default='./split_mnist_grad_cosine')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5

    os.makedirs(args.log_dir, exist_ok=True)

    model = MLPModel(hidden_dim=args.hidden_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    print("=" * 50)
    print("MLP + IntervalActivation + GradientCosine on Split MNIST")
    print(f"Hidden dim: {args.hidden_dim}, device: {device}, real_bounds: {args.real_bounds}")
    print("=" * 50)

    # ---- Task 0 ----
    print("\n[Task 0] Training on digits 0-4...")
    train_0, test_0 = get_split_mnist(0, args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs0):
        model.train()
        total_loss = 0.0
        for xb, yb in train_0:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb, train=True)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, test_0, device)
        print(f"  Epoch {epoch + 1}: loss={total_loss / len(train_0):.4f}, test acc={acc:.4f}")

    # ---- Fill IA buffers, compute bounds ----
    print("[Task 0] Filling IA test_act_buffer for bound estimation...")
    fill_ia_buffers(model, train_0, device)

    interval_pen = IntervalPenalization(
        var_loss_scale=0.01,
        internal_repr_drift_loss_scale=0.001,
        feature_loss_scale=0.1,
        use_align_loss=True,
    ).to(device)

    print("[Interval] setup_task(0)...")
    interval_pen.setup_task(0, model.classifier, model.feature_extractor, model.feature_extractor)

    # ---- Gradient cosine tracker ----
    csv_path = os.path.join(args.log_dir, "grad_cosine.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    grad_tracker = GradientCosineTracker(opt, csv_path)

    # ---- Task 1 setup ----
    print("\n[Task 1] setup_task(1) — snapshots old params, computes cumulative bounds...")
    interval_pen.setup_task(1, model.classifier, model.feature_extractor, model.feature_extractor)
    interval_pen.gradient_tracker = grad_tracker

    if not args.real_bounds:
        set_artificial_bounds(model)
        print("  Using artificial bounds (split-dims) to force both loss gradients non-zero")

    # ---- Task 1 training ----
    print("[Task 1] Training on digits 5-9 with drift + align losses...")
    train_1, test_1 = get_split_mnist(1, args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_1:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb, train=True)
            ce_loss = criterion(logits, yb)
            total = interval_pen.forward(xb, ce_loss)
            opt.zero_grad()
            total.backward(retain_graph=True)
            opt.step()
            total_loss += total.item()
        acc = evaluate(model, test_1, device)
        print(f"  Epoch {epoch + 1}: loss={total_loss / len(train_1):.4f}, test acc={acc:.4f}")

    print(f"\nDone. Gradient cosine logged to {csv_path}")
    print_cosine_summary(csv_path)


if __name__ == '__main__':
    main()

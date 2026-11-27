"""Train a quick SignBridge sequence classifier from extracted/training_dataset.npz.

Important: each class currently has one source video. This script trains a demo
prototype and deliberately does NOT report a validation/test accuracy.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, X, y, frame_mask, pose_ok, left_ok, right_ok, augment=False):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.frame_mask = torch.from_numpy(frame_mask.astype(bool))
        self.pose_ok = torch.from_numpy(pose_ok.astype(bool))
        self.left_ok = torch.from_numpy(left_ok.astype(bool))
        self.right_ok = torch.from_numpy(right_ok.astype(bool))
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].clone()
        valid = self.frame_mask[i]
        if self.augment:
            # Small coordinate noise only on detected landmark groups.
            for mask, start, end in ((self.pose_ok[i], 0, 132),
                                     (self.left_ok[i], 132, 195),
                                     (self.right_ok[i], 195, 258)):
                if mask.any():
                    noise = torch.randn_like(x[:, start:end]) * 0.008
                    x[mask, start:end] += noise[mask]
            # Mild temporal jitter: duplicate/drop a few interior frames.
            if valid.sum() > 12 and random.random() < 0.35:
                n = int(valid.sum())
                shift = random.choice([-2, -1, 1, 2])
                seq = x[:n].clone()
                if shift > 0:
                    seq[shift:] = seq[:-shift].clone()
                else:
                    seq[:shift] = seq[-shift:].clone()
                x[:n] = seq
            # Randomly hide a short time span, retaining at least most frames.
            if valid.sum() > 20 and random.random() < 0.25:
                n = int(valid.sum())
                span = random.randint(1, max(1, n // 12))
                start = random.randint(0, max(0, n - span))
                x[start:start + span] = 0
        return x, self.y[i]


class SignGRU(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(258)
        self.gru = nn.GRU(input_size=258, hidden_size=128, num_layers=2,
                          batch_first=True, dropout=0.15, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 192), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        _, h = self.gru(x)
        # Last layer's forward and backward states.
        pooled = torch.cat((h[-2], h[-1]), dim=1)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='extracted/training_dataset.npz')
    ap.add_argument('--out', default='training_output')
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.is_file(): raise SystemExit(f'Dataset tidak ditemukan: {data_path}')
    with np.load(data_path, allow_pickle=False) as d:
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.int64)
        names = d['class_names'].astype(str).tolist()
        frame_mask = d['frame_mask']
        pose_ok, left_ok, right_ok = d['pose_ok'], d['left_ok'], d['right_ok']

    if X.ndim != 3 or X.shape[1:] != (128, 258):
        raise SystemExit(f'Bentuk X tidak sesuai; ditemukan {X.shape}, seharusnya (N,128,258)')
    if len(X) != len(y) or len(names) != len(set(names)):
        raise SystemExit('Jumlah data/label tidak konsisten.')
    if not np.isfinite(X).all(): raise SystemExit('Dataset mengandung NaN/Inf.')
    if len(np.unique(y)) != len(names): raise SystemExit('ID label tidak cocok dengan class_names.')

    # One source recording per class: train on all available examples with
    # online augmentation. No misleading random frame split or accuracy score.
    ds = SequenceDataset(X, y, frame_mask, pose_ok, left_ok, right_ok, augment=True)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignGRU(len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb); total += len(yb)
        scheduler.step()
        mean_loss = total_loss / max(total, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({'model_state_dict': model.state_dict(), 'num_classes': len(names),
                        'input_shape': [128, 258]}, out / 'signbridge_gru.pt')
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch:03d}/{args.epochs} | train_loss={mean_loss:.5f}')

    (out / 'class_names.json').write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {
        'architecture': '2-layer bidirectional GRU', 'classes': len(names),
        'sequence_length': 128, 'features_per_frame': 258,
        'source_videos_per_class': 1,
        'best_training_loss': best_loss,
        'device': str(device),
        'warning': 'Training loss is not validation accuracy. One source video per class; real-world generalization is unverified.'
    }
    (out / 'training_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSELESAI. Model:', out / 'signbridge_gru.pt')
    print('Kelas:', len(names), '| Best train loss:', round(best_loss, 5))
    print('CATATAN: tidak ada validation accuracy karena hanya satu video sumber per kelas.')

if __name__ == '__main__': main()

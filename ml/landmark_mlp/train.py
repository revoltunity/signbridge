"""Training MLP 25 kelas dari landmark; split grup agar satu grup tidak bocor antar-split."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from common import FEATURE_DIM, featurize, mirror


def feats(L, n, flip=False):
    return np.asarray([featurize(mirror(l, k) if flip else l, k) for l, k in zip(L, n)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="landmarks.npz")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with np.load(a.npz, allow_pickle=False) as z:
        if "ver" not in z.files or int(z["ver"]) != 3:
            raise SystemExit("landmarks.npz bukan format v3. Jalankan ulang extract.py versi terbaru.")
        L, n, cls, grp = z["L"], z["n"], z["cls"], z["grp"]
        classes = [str(x) for x in z["class_names"]]
    if not len(cls):
        raise SystemExit("NPZ kosong: tidak ada landmark yang berhasil diekstrak.")
    if L.shape != (len(cls), 2, 21, 3) or len(n) != len(cls) or len(grp) != len(cls):
        raise SystemExit("Struktur landmarks.npz tidak konsisten.")
    keep = n > 0
    print(f"Gambar tersimpan: {len(keep)} | tangan terdeteksi: {int(keep.sum())} ({keep.mean():.1%}) | tanpa tangan: {int((~keep).sum())}")
    L, n, cls, grp = L[keep], n[keep], cls[keep], grp[keep]

    absent = [c for c in classes if not np.any(cls == c)]
    if absent:
        raise SystemExit("Kelas tanpa satu pun foto bertangan terdeteksi: " + ", ".join(absent) + ". Periksa foto/posisi tangan.")
    insufficient = []
    groups_by_class = {}
    for c in classes:
        gs = sorted(set(grp[cls == c].tolist()))
        groups_by_class[c] = gs
        if len(gs) < 3:
            insufficient.append(f"{c}={len(gs)}")
    if insufficient:
        raise SystemExit("Minimal 3 grup per kelas diperlukan untuk train/val/test. Grup: " + ", ".join(insufficient) + ". Tinjau --block-size saat ekstraksi.")

    rng = random.Random(a.seed)
    part = {}
    for c in classes:
        gs = groups_by_class[c].copy(); rng.shuffle(gs)
        nt = max(1, round(len(gs) * 0.15)); nv = max(1, round(len(gs) * 0.15))
        if nt + nv >= len(gs):
            nt, nv = 1, 1
        for i, g in enumerate(gs):
            part[g] = "test" if i < nt else "val" if i < nt + nv else "train"
    split = np.asarray([part[g] for g in grp])
    for s in ("train", "val", "test"):
        mask = split == s
        print(f"{s:5s}: {int(mask.sum()):5d} foto | {len(set(grp[mask])):4d} grup | {len(set(cls[mask])):2d}/{len(classes)} kelas")
        missing = [c for c in classes if not np.any(cls[mask] == c)]
        if missing:
            raise SystemExit(f"Split {s} kehilangan kelas: {', '.join(missing)}")

    ci = {c: i for i, c in enumerate(classes)}
    y = np.asarray([ci[str(c)] for c in cls], dtype=np.int64)
    tr, va, te = split == "train", split == "val", split == "test"
    Xtr, ytr = feats(L[tr], n[tr]), y[tr]
    if not a.no_mirror:
        Xtr = np.vstack((Xtr, feats(L[tr], n[tr], True)))
        ytr = np.concatenate((ytr, ytr))
    Xva, Xte = feats(L[va], n[va]), feats(L[te], n[te])
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-6] = 1.0
    norm = lambda x: (x - mu) / sd
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=a.seed)
    clf.fit(norm(Xtr), ytr)

    report = {}
    predictions = {}
    for name, X, yy in (("val", Xva, y[va]), ("test", Xte, y[te])):
        pred = clf.predict(norm(X)); predictions[name] = pred
        report[name] = float(accuracy_score(yy, pred))
        print(f"\n== {name}: akurasi {report[name]:.4f} ({len(yy)} foto)")
    print("\nLaporan per kelas (test):")
    print(classification_report(y[te], predictions["test"], labels=range(len(classes)), target_names=classes, zero_division=0))
    cm = confusion_matrix(y[te], predictions["test"], labels=range(len(classes)))
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    pairs = sorted(((int(cm_off[i, j]), classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if cm_off[i, j]), reverse=True)[:15]
    print("Salah terbanyak (benar -> prediksi):", [f"{x}->{y}:{count}" for count, x, y in pairs] or "tidak ada")

    model = {
        "classes": classes, "feature_dim": FEATURE_DIM, "mean": mu.tolist(), "std": sd.tolist(),
        "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(clf.coefs_, clf.intercepts_)],
        "activation": "relu", "output": "softmax", "val_acc": report["val"], "test_acc": report["test"],
    }
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    (out / "report.json").write_text(json.dumps({
        "classes": classes, "val": report["val"], "test": report["test"],
        "top_errors": [[count, actual, predicted] for count, actual, predicted in pairs],
        "split_groups": {s: sorted(set(grp[split == s].tolist())) for s in ("train", "val", "test")},
        "warning": "Akurasi bergantung pada validitas grup sesi; blok nomor file bukan bukti sesi pengambilan sebenarnya.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTersimpan: {out / 'model.json'}")
    pub = Path(__file__).resolve().parents[2] / "web" / "public" / "models"
    if pub.is_dir():
        (pub / "hand_mlp.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
        print(f"Disalin ke web: {pub / 'hand_mlp.json'}")


if __name__ == "__main__":
    main()

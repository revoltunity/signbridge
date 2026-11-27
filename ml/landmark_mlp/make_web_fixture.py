"""Membuat web/tests/fixtures/handmlp.json: model kecil sintetis + fitur & probabilitas referensi dari Python,
untuk memastikan implementasi JS (handMlp.js) identik dengan common.py + sklearn."""
import json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from common import featurize, FEATURE_DIM

rng = np.random.default_rng(1)
C = ["A", "B", "C"]; W, H = 1280, 720
base = [rng.normal(size=(21, 3)) * 0.04 for _ in C]

def make(c, two):
    hands = []
    for k in range(2 if two else 1):
        ctr = np.array([0.3 + 0.35 * k, 0.5, 0.0]) + rng.normal(size=3) * 0.02
        hands.append(ctr + base[c] + rng.normal(size=(21, 3)) * 0.004)
    return hands

def to_L(hands):
    L = np.zeros((2, 21, 3))
    for i, h in enumerate(hands): L[i] = h * np.array([W, H, W])
    return L

X, y = [], []
for i in range(300):
    c = i % 3; hs = make(c, i % 5 == 0)
    X.append(featurize(to_L(hs), len(hs))); y.append(c)
X = np.array(X); mu, sd = X.mean(0), X.std(0) + 1e-6
clf = MLPClassifier((16, 8), max_iter=400, random_state=0).fit((X - mu) / sd, y)
model = {"classes": C, "feature_dim": FEATURE_DIM, "mean": mu.tolist(), "std": sd.tolist(),
         "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(clf.coefs_, clf.intercepts_)]}
cases = []
for c, two in [(0, False), (1, False), (2, False), (0, True), (1, True)]:
    hs = make(c, two); f = featurize(to_L(hs), len(hs))
    cases.append({"hands": [h.tolist() for h in hs], "width": W, "height": H, "label": c,
                  "features": f.tolist(), "probs": clf.predict_proba(((f - mu) / sd)[None])[0].tolist()})
out = Path(__file__).resolve().parents[2] / "web" / "tests" / "fixtures" / "handmlp.json"
out.write_text(json.dumps({"model": model, "cases": cases})); print("OK", out)

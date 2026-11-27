"""Fitur landmark tangan bersama untuk ekstraksi dan inferensi web."""
import numpy as np

FEATURE_DIM = 131


def featurize(L, n):
    """L: (2, 21, 3), koordinat x/y/z sudah dalam piksel; n: 0..2."""
    L = np.asarray(L, dtype=np.float64)
    n = int(n)
    if L.shape != (2, 21, 3):
        raise ValueError(f"Bentuk L harus (2,21,3), didapat {L.shape}")
    if not 0 <= n <= 2:
        raise ValueError(f"Jumlah tangan harus 0..2, didapat {n}")
    if not np.isfinite(L).all():
        raise ValueError("Landmark mengandung NaN/Inf")

    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    hands = [L[i].copy() for i in range(n)]
    hands.sort(key=lambda h: h[0, 0])
    scales = []
    for i, h in enumerate(hands):
        p = h - h[0]
        s = float(np.linalg.norm(p[9, :2]))
        if s < 1e-6:
            s = 1e-6
        scales.append(s)
        f[i * 63:(i + 1) * 63] = (p / s).reshape(-1)
        f[126 + i] = 1.0
    if n == 2:
        f[128:131] = (hands[1][0] - hands[0][0]) / scales[0]
    return f


def mirror(L, n):
    """Cermin horizontal; urutan tangan akan dinormalisasi ulang oleh featurize."""
    M = np.asarray(L).copy()
    M[:int(n), :, 0] *= -1
    return M

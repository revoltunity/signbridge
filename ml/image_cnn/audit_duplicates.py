
from pathlib import Path
from PIL import Image
import numpy as np
import csv

ROOT = Path(r"C:\AL KAUTSAR\DATA YAZID\LOMBA\datasetkata\IsyaratKata")
OUTPUT = ROOT.parent / "audit_duplikat.csv"
THRESHOLD = 6

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def dhash(path):
    with Image.open(path) as img:
        img = img.convert("L").resize((9, 8))
        pixels = np.asarray(img)
        bits = pixels[:, 1:] > pixels[:, :-1]

        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)

        return value

files = []
hashes = []
failed = []

print("Membaca dan menghitung perceptual hash...")

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            files.append((folder.name, path.name, str(path)))
            hashes.append(dhash(path))
        except Exception as e:
            failed.append((str(path), str(e)))

print("Gambar berhasil diproses:", len(files))
print("Gambar gagal:", len(failed))

# Perbandingan seluruh gambar, termasuk antar kelas
arr = np.array(hashes, dtype=np.uint64)
lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

results = []
block_size = 128

print("Mencari gambar yang mirip...")

for start in range(0, len(arr), block_size):
    end = min(start + block_size, len(arr))

    xor = np.bitwise_xor(arr[start:end, None], arr[None, :])
    byte_values = xor.view(np.uint8).reshape(end-start, len(arr), 8)

    distances = lookup[byte_values].sum(axis=2)

    for local_i in range(end-start):
        i = start + local_i

        # Hanya membandingkan pasangan sekali
        distances[local_i, :i+1] = 255

    row_idx, col_idx = np.where(distances <= THRESHOLD)

    for i, j in zip(row_idx, col_idx):
        a = files[start + i]
        b = files[j]

        results.append({
            "kelas_1": a[0],
            "file_1": a[1],
            "kelas_2": b[0],
            "file_2": b[1],
            "jarak_hash": int(distances[i, j]),
            "kemiripan_hash_persen": round(
                (64 - int(distances[i, j])) / 64 * 100, 2
            ),
            "path_1": a[2],
            "path_2": b[2]
        })

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    columns = [
        "kelas_1", "file_1", "kelas_2", "file_2",
        "jarak_hash", "kemiripan_hash_persen",
        "path_1", "path_2"
    ]
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

print("\n===== HASIL AUDIT =====")
print("Total gambar:", len(files))
print("Pasangan mencurigakan:", len(results))
print("Gambar gagal:", len(failed))
print("Laporan:", OUTPUT)

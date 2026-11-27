from pathlib import Path
import json
import numpy as np

SOURCE = Path("extracted/landmarks")
OUTPUT = Path("extracted/processed")
EXCLUDED = Path("quarantine/excluded_labels.txt")

OUTPUT.mkdir(parents=True, exist_ok=True)
excluded = set(EXCLUDED.read_text(encoding="utf-8").splitlines()) if EXCLUDED.exists() else set()

files = sorted(SOURCE.glob("*.npz"))
processed = 0
errors = []

for file in files:
    label = file.stem
    if label in excluded:
        continue

    try:
        with np.load(file, allow_pickle=False) as d:
            pose = d["pose"].copy()
            left = d["left_hand"].copy()
            right = d["right_hand"].copy()
            pose_ok = d["pose_ok"].copy()
            left_ok = d["left_ok"].copy()
            right_ok = d["right_ok"].copy()

            # Koordinat x dan y sudah ternormalisasi oleh MediaPipe.
            # Jangan dibagi lagi dengan ukuran frame.
            # Pertahankan z, visibility, dan mask deteksi.

            pose[~pose_ok] = 0
            left[~left_ok] = 0
            right[~right_ok] = 0

            np.savez_compressed(
                OUTPUT / f"{label}.npz",
                pose=pose,
                left_hand=left,
                right_hand=right,
                pose_ok=pose_ok,
                left_ok=left_ok,
                right_ok=right_ok,
                frame_idx=d["frame_idx"],
                timestamp=d["timestamp"],
                label=np.array(label)
            )
        processed += 1
    except Exception as e:
        errors.append((label, str(e)))

print("FILE DIPROSES:", processed)
print("FILE ERROR:", len(errors))
for item in errors:
    print(*item)
print("OUTPUT:", OUTPUT)

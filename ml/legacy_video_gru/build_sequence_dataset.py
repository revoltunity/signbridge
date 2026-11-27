from pathlib import Path
import glob
import numpy as np

SOURCE = Path("extracted/processed")
OUTPUT = Path("extracted/training_dataset.npz")
SEQ_LEN = 128

files = sorted(glob.glob(str(SOURCE / "*.npz")))
if not files:
    raise RuntimeError("Tidak ada file preprocessing.")

labels = [Path(f).stem for f in files]
class_names = sorted(labels)
class_to_id = {name: i for i, name in enumerate(class_names)}

N = len(files)
X = np.zeros((N, SEQ_LEN, 258), dtype=np.float32)
pose_ok_all = np.zeros((N, SEQ_LEN), dtype=bool)
left_ok_all = np.zeros((N, SEQ_LEN), dtype=bool)
right_ok_all = np.zeros((N, SEQ_LEN), dtype=bool)
frame_mask = np.zeros((N, SEQ_LEN), dtype=bool)
timestamps = np.zeros((N, SEQ_LEN), dtype=np.float32)
y = np.zeros(N, dtype=np.int64)

for i, file in enumerate(files):
    with np.load(file, allow_pickle=False) as d:
        n = len(d["frame_idx"])

        if n > SEQ_LEN:
            indices = np.linspace(0, n - 1, SEQ_LEN).round().astype(int)
        else:
            indices = np.arange(n)

        length = len(indices)

        pose = d["pose"][indices].reshape(length, -1)
        left = d["left_hand"][indices].reshape(length, -1)
        right = d["right_hand"][indices].reshape(length, -1)

        X[i, :length] = np.concatenate([pose, left, right], axis=1)
        pose_ok_all[i, :length] = d["pose_ok"][indices]
        left_ok_all[i, :length] = d["left_ok"][indices]
        right_ok_all[i, :length] = d["right_ok"][indices]
        frame_mask[i, :length] = True
        timestamps[i, :length] = d["timestamp"][indices]
        y[i] = class_to_id[Path(file).stem]

    if (i + 1) % 25 == 0 or i + 1 == N:
        print(f"Memproses: {i + 1}/{N}")

np.savez_compressed(
    OUTPUT,
    X=X,
    y=y,
    class_names=np.array(class_names),
    pose_ok=pose_ok_all,
    left_ok=left_ok_all,
    right_ok=right_ok_all,
    frame_mask=frame_mask,
    timestamps=timestamps
)

print("\nSELESAI")
print("Jumlah video:", N)
print("Jumlah kelas:", len(class_names))
print("Bentuk X:", X.shape)
print("Panjang sequence:", SEQ_LEN)
print("Output:", OUTPUT)

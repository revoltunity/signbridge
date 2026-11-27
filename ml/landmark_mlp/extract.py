
"""Ekstraksi foto -> landmark MediaPipe.
Grup dibuat dari blok nomor foto dan penggabungan blok yang memiliki
file identik berdasarkan SHA256. Tidak menghapus atau mengubah foto.
"""
import argparse
import hashlib
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VER = 4


def numeric_id(path: Path):
    """Ambil nomor utama; abaikan akhiran duplikat seperti (1), (2)."""
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_groups(all_images, class_names, block_size, first_number):
    """Gabungkan blok yang berisi foto identik; variasi foto tetap terpisah."""
    parent = {}
    image_keys = {}
    hashes_by_class = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Pilih root stabil agar ID grup konsisten.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    print("Menghitung SHA256 dan menyusun grup foto...", flush=True)

    for i, (class_name, path) in enumerate(all_images, 1):
        number = numeric_id(path)

        if number is not None and first_number[class_name] is not None:
            block = (number - first_number[class_name]) // block_size
            key = f"{class_name}:block:{block}"
        else:
            key = f"{class_name}:name:{path.stem}"

        find(key)
        rel = path.relative_to(DATA_ROOT).as_posix()
        image_keys[rel] = key

        digest = sha256_file(path)
        hash_key = (class_name, digest)
        hashes_by_class.setdefault(hash_key, []).append(key)

        if i % 500 == 0 or i == len(all_images):
            print(f"Hash {i}/{len(all_images)}", flush=True)

    duplicate_groups = 0
    duplicate_blocks = 0

    for keys in hashes_by_class.values():
        unique_keys = sorted(set(keys))
        if len(keys) > 1:
            duplicate_groups += 1
            duplicate_blocks += len(unique_keys)
            for key in unique_keys[1:]:
                union(unique_keys[0], key)

    result = {}
    for rel, key in image_keys.items():
        result[rel] = find(key)

    print(
        f"Grup hash duplikat: {duplicate_groups} | "
        f"blok yang terhubung: {duplicate_blocks}",
        flush=True,
    )
    return result


def save(out, files, sizes, mtimes, Ls, ns, cls, groups, class_names):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp.npz")

    np.savez_compressed(
        tmp,
        ver=np.array(VER),
        files=np.asarray(files, dtype=str),
        sizes=np.asarray(sizes, dtype=np.int64),
        mtimes=np.asarray(mtimes, dtype=np.int64),
        L=np.asarray(Ls, dtype=np.float32).reshape((-1, 2, 21, 3)),
        n=np.asarray(ns, dtype=np.int8),
        cls=np.asarray(cls, dtype=str),
        grp=np.asarray(groups, dtype=str),
        class_names=np.asarray(class_names, dtype=str),
    )
    tmp.replace(out)


def main():
    global DATA_ROOT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Folder berisi subfolder per kelas")
    ap.add_argument("--out", default="landmarks.npz")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=500)
    a = ap.parse_args()

    if a.max_side < 64 or a.block_size < 1 or a.save_every < 1:
        ap.error("--max-side minimal 64; --block-size dan --save-every harus >= 1")

    root = Path(a.data).expanduser().resolve()
    DATA_ROOT = root

    if not root.is_dir():
        raise SystemExit(f"Folder dataset tidak ditemukan: {root}")

    class_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not class_dirs:
        raise SystemExit(f"Tidak ada subfolder kelas di: {root}")

    class_names = [p.name for p in class_dirs]
    all_images = [
        (c.name, p)
        for c in class_dirs
        for p in sorted(c.rglob("*"))
        if p.is_file() and p.suffix.lower() in EXT
    ]

    if not all_images:
        raise SystemExit("Tidak ada gambar JPG/JPEG/PNG/BMP/WEBP.")

    first_number = {}
    for c in class_names:
        nums = [
            numeric_id(p)
            for label, p in all_images
            if label == c
        ]
        nums = [x for x in nums if x is not None]
        first_number[c] = min(nums) if nums else None

    groups_by_file = build_groups(
        all_images, class_names, a.block_size, first_number
    )

    no_numeric = [
        c for c, first in first_number.items()
        if first is None
    ]
    if no_numeric:
        print(
            "PERINGATAN: kelas tanpa nomor file:",
            ", ".join(no_numeric),
        )

    out = Path(a.out).resolve()
    cache = {}

    if out.exists():
        try:
            with np.load(out, allow_pickle=False) as z:
                if "ver" in z.files and int(z["ver"]) == VER:
                    for i, rel in enumerate(z["files"]):
                        cache[str(rel)] = (
                            int(z["sizes"][i]),
                            int(z["mtimes"][i]),
                            z["L"][i].copy(),
                            int(z["n"][i]),
                            str(z["grp"][i]),
                        )
                    print(
                        f"Resume: {len(cache)} rekaman tersimpan."
                    )
                else:
                    print("Cache versi lama: ekstraksi ulang dilakukan.")
        except Exception as e:
            print(f"Cache tidak dapat dibaca ({e}); ekstraksi ulang.")

    import mediapipe as mp
    from PIL import Image, ImageOps
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    task = Path(__file__).resolve().parent / "hand_landmarker.task"
    if not task.exists():
        print("Mengunduh model tangan MediaPipe...")
        urllib.request.urlretrieve(URL, task)

    options = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    files, sizes, mtimes = [], [], []
    Ls, ns, cls, groups = [], [], [], []
    failed = {}
    t0 = time.time()

    with vision.HandLandmarker.create_from_options(options) as det:
        for k, (c, p) in enumerate(all_images, 1):
            rel = p.relative_to(root).as_posix()
            st = p.stat()
            size, mtime = st.st_size, st.st_mtime_ns
            group = groups_by_file[rel]

            old = cache.get(rel)

            if (
                old
                and old[0] == size
                and old[1] == mtime
                and old[4] == group
            ):
                L, n = old[2], old[3]
            else:
                try:
                    with Image.open(p) as source:
                        im = ImageOps.exif_transpose(source).convert("RGB")

                    im.thumbnail((a.max_side, a.max_side))
                    result = det.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.asarray(im),
                        )
                    )

                    L = np.zeros((2, 21, 3), dtype=np.float32)
                    n = min(len(result.hand_landmarks), 2)

                    for i in range(n):
                        L[i] = [
                            [q.x * im.width, q.y * im.height, q.z * im.width]
                            for q in result.hand_landmarks[i]
                        ]

                except Exception as e:
                    failed.setdefault(c, []).append(f"{rel}: {e}")
                    continue

            files.append(rel)
            sizes.append(size)
            mtimes.append(mtime)
            Ls.append(L)
            ns.append(n)
            cls.append(c)
            groups.append(group)

            if k % 100 == 0 or k == len(all_images):
                print(
                    f"{k}/{len(all_images)} gambar | "
                    f"{time.time()-t0:.0f}s | {c}",
                    flush=True,
                )

            if k % a.save_every == 0:
                save(out, files, sizes, mtimes, Ls, ns, cls, groups, class_names)

    save(out, files, sizes, mtimes, Ls, ns, cls, groups, class_names)

    n_arr = np.asarray(ns)
    c_arr = np.asarray(cls)
    grp_arr = np.asarray(groups)

    print(f"\nSelesai: {len(files)}/{len(all_images)} gambar tersimpan -> {out}")
    print("Grup: blok nomor foto + penggabungan blok dengan foto identik SHA256.")
    print("Catatan: blok nomor tetap perkiraan, bukan bukti sesi pengambilan.")

    for c in class_names:
        mask = c_arr == c
        total = int(mask.sum())
        no_hand = int((n_arr[mask] == 0).sum()) if total else 0
        groups_count = len(set(grp_arr[mask])) if total else 0

        if total:
            print(
                f"{c:16s} total={total:5d} "
                f"tanpa_tangan={no_hand:5d} ({no_hand/total:.1%}) "
                f"grup={groups_count}"
            )
        else:
            print(f"{c:16s} TOTAL=0 (tidak ada gambar berhasil diproses)")

    if failed:
        print("\nGambar gagal dibaca:", {c: len(v) for c, v in failed.items()})
        print("Contoh:", next(iter(next(iter(failed.values()))), ""))


if __name__ == "__main__":
    main()

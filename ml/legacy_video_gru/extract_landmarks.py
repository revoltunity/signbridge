#!/usr/bin/env python3
"""SignBridge - ekstraksi landmark pose + dua tangan dari seluruh video SIBI.

Prinsip (sesuai handover):
- SEMUA frame diproses (tanpa membuang detik pertama, tanpa membuang frame tanpa tangan).
- Landmark hilang = NaN + mask (present) eksplisit, BUKAN nol.
- Video asli tidak diubah. Duplikat byte-identik dilewati (dicatat di laporan).
- Resume: video yang sudah punya .npz valid dengan parameter sama dilewati.
- Video gagal dicatat, proses lanjut.

Output per video: <out>/landmarks/<label>.npz berisi
  pose        (T,33,4)  x,y,z,visibility  (x,y ternormalisasi 0-1 terhadap frame)
  left_hand   (T,21,3)  x,y,z
  right_hand  (T,21,3)
  pose_ok     (T,) bool
  left_ok     (T,) bool
  right_ok    (T,) bool
  frame_idx   (T,) int
  timestamp   (T,) float detik
  meta        JSON string (label, file, fps, w, h, config)
Laporan: <out>/reports/video_quality.csv, run_summary.json, duplicates.csv

Catatan handedness: MediaPipe Hands memberi label "Left/Right" dengan asumsi gambar mirror.
Video kamus bukan selfie-mirror, jadi label bisa terbalik. Karena itu tangan ditetapkan ke slot
kiri/kanan berdasarkan POSISI relatif terhadap bahu (pose) bila pose tersedia, label MediaPipe
hanya sebagai cadangan. Lihat assign_hands().
"""
import argparse, csv, hashlib, json, re, sys, time, traceback
from pathlib import Path
import numpy as np

CONFIG_VERSION = 1
DEFAULTS = dict(det_conf=0.3, track_conf=0.1, hand_complexity=0, pose_complexity=1)


def file_md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def label_from_name(name):
    """'A (1).webm' -> 'A'; 'Awalan-Di.webm' -> 'Awalan-Di'. Label = nama file tanpa akhiran salinan."""
    stem = Path(name).stem
    return re.sub(r"\s*\(\d+\)$", "", stem).strip()


def category_from_label(label):
    l = label.casefold()
    if l.startswith("awalan-"): return "awalan"
    if l.startswith("akhiran-"): return "akhiran"
    if l.startswith("partikel-"): return "partikel"
    if re.fullmatch(r"[a-z]", l): return "alfabet"
    if re.fullmatch(r"\d+", l): return "angka"
    return "kata"


def assign_hands(hand_lms, handedness, pose_xy):
    """Kembalikan (left, right) masing-masing (21,3) atau None.
    Dua tangan: yang x-nya lebih kecil di gambar -> 'kanan subjek' bila subjek menghadap kamera
    (tanpa mirror). Dipakai konsisten untuk semua video; yang penting KONSISTEN, bukan label anatomi.
    Satu tangan: dekati bahu terdekat (pose) bila ada, kalau tidak pakai x relatif ke tengah frame.
    Slot 'left_hand' = sisi kiri GAMBAR, 'right_hand' = sisi kanan GAMBAR.
    """
    if not hand_lms:
        return None, None
    arr = [np.array([[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32) for h in hand_lms]
    if len(arr) >= 2:
        arr = sorted(arr[:2], key=lambda a: a[0, 0])
        return arr[0], arr[1]
    a = arr[0]
    if pose_xy is not None:
        ls, rs = pose_xy  # bahu kiri/kanan GAMBAR (x kecil, x besar)
        mid = (ls[0] + rs[0]) / 2
    else:
        mid = 0.5
    return (a, None) if a[0, 0] < mid else (None, a)


def process_video(path, hands, pose, every):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("tidak bisa dibuka")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    P, L, R, pok, lok, rok, fidx, ts = [], [], [], [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pr = pose.process(rgb)
            hr = hands.process(rgb)
            if pr.pose_landmarks:
                pl = np.array([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks.landmark], dtype=np.float32)
                pok.append(True)
                # bahu GAMBAR kiri/kanan (indeks pose 11 dan 12)
                s = sorted([pl[11, :2], pl[12, :2]], key=lambda q: q[0])
                shoulders = (s[0], s[1])
            else:
                pl = np.full((33, 4), np.nan, np.float32); pok.append(False); shoulders = None
            l, r = assign_hands(hr.multi_hand_landmarks, hr.multi_handedness, shoulders)
            L.append(l if l is not None else np.full((21, 3), np.nan, np.float32)); lok.append(l is not None)
            R.append(r if r is not None else np.full((21, 3), np.nan, np.float32)); rok.append(r is not None)
            P.append(pl); fidx.append(i); ts.append(i / fps)
        i += 1
    cap.release()
    if not P:
        raise RuntimeError("0 frame terbaca")
    return dict(pose=np.stack(P), left_hand=np.stack(L), right_hand=np.stack(R),
                pose_ok=np.array(pok), left_ok=np.array(lok), right_ok=np.array(rok),
                frame_idx=np.array(fidx, np.int32), timestamp=np.array(ts, np.float32)), fps, w, h, i


def quality_row(file, label, status, d=None, fps=None, w=None, h=None, decoded=None, secs=None, err="", dup_of=""):
    row = dict(file=file, label=label, category=category_from_label(label), status=status,
               duplicate_of=dup_of, error=err)
    if d is not None:
        T = len(d["frame_idx"])
        any_hand = d["left_ok"] | d["right_ok"]
        both = d["left_ok"] & d["right_ok"]
        # run terpanjang frame berurutan tanpa tangan (indikator lubang temporal)
        gap = best = 0
        for v in any_hand:
            gap = 0 if v else gap + 1
            best = max(best, gap)
        row.update(frames=T, decoded_frames=decoded, fps=round(fps, 3), width=w, height=h,
                   pose_pct=round(100 * d["pose_ok"].mean(), 2),
                   hand0_pct=round(100 * (~any_hand).mean(), 2),
                   hand1_pct=round(100 * (any_hand & ~both).mean(), 2),
                   hand2_pct=round(100 * both.mean(), 2),
                   left_pct=round(100 * d["left_ok"].mean(), 2),
                   right_pct=round(100 * d["right_ok"].mean(), 2),
                   longest_no_hand_gap=best, proc_sec=round(secs, 2))
        flags = []
        if T < 16: flags.append("SHORT")
        if row["pose_pct"] < 90: flags.append("LOW_POSE")
        if row["hand0_pct"] > 60: flags.append("MOSTLY_NO_HAND")
        if best > 0.5 * T: flags.append("LONG_HAND_GAP")
        row["flags"] = "|".join(flags)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Folder berisi .webm")
    ap.add_argument("--out", default="extracted", help="Folder output (default: extracted)")
    ap.add_argument("--every", type=int, default=1, help="Ambil setiap N frame (default 1 = semua)")
    ap.add_argument("--det-conf", type=float, default=DEFAULTS["det_conf"])
    ap.add_argument("--track-conf", type=float, default=DEFAULTS["track_conf"])
    ap.add_argument("--hand-complexity", type=int, default=DEFAULTS["hand_complexity"], choices=[0, 1])
    ap.add_argument("--pose-complexity", type=int, default=DEFAULTS["pose_complexity"], choices=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah video (uji cepat)")
    ap.add_argument("--force", action="store_true", help="Proses ulang walau sudah ada")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Folder tidak ditemukan: {root}")
    out = Path(args.out).expanduser().resolve()
    (out / "landmarks").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    cfg = dict(version=CONFIG_VERSION, every=args.every, det_conf=args.det_conf, track_conf=args.track_conf,
               hand_complexity=args.hand_complexity, pose_complexity=args.pose_complexity)
    files = sorted(root.glob("*.webm"), key=lambda p: p.name.casefold())
    if args.limit: files = files[:args.limit]
    print(f"{len(files)} video ditemukan di {root}")

    # --- duplikat byte-identik: simpan satu (pilih nama tanpa '(n)'), lewati sisanya ---
    by_hash, dup_rows = {}, []
    for p in files:
        by_hash.setdefault(file_md5(p), []).append(p)
    skip = {}
    for hsh, ps in by_hash.items():
        if len(ps) > 1:
            ps = sorted(ps, key=lambda p: (bool(re.search(r"\(\d+\)", p.name)), p.name.casefold()))
            for extra in ps[1:]:
                skip[extra.name] = ps[0].name
                dup_rows.append(dict(kept=ps[0].name, skipped=extra.name, md5=hsh))
    with (out / "reports" / "duplicates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["kept", "skipped", "md5"]); w.writeheader(); w.writerows(dup_rows)

    # label ganda (setelah normalisasi) selain duplikat byte -> peringatan
    seen = {}
    for p in files:
        if p.name in skip: continue
        seen.setdefault(label_from_name(p.name).casefold(), []).append(p.name)
    for lab, ns in seen.items():
        if len(ns) > 1: print(f"PERINGATAN label ganda '{lab}': {ns}")

    import cv2, mediapipe as mp
    if not hasattr(mp, "solutions"):
        sys.exit("mp.solutions tidak ada. Pasang: python -m pip install mediapipe==0.10.21")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     model_complexity=args.hand_complexity,
                                     min_detection_confidence=args.det_conf,
                                     min_tracking_confidence=args.track_conf)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=args.pose_complexity,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    rows, t0 = [], time.time()
    for n, p in enumerate(files, 1):
        label = label_from_name(p.name)
        npz = out / "landmarks" / f"{label}.npz"
        rowf = out / "reports" / "rows" / f"{p.stem}.json"
        rowf.parent.mkdir(exist_ok=True)
        if p.name in skip:
            rows.append(quality_row(p.name, label, "SKIPPED_DUPLICATE", dup_of=skip[p.name]))
            print(f"[{n}/{len(files)}] {p.name}: duplikat dari {skip[p.name]}, dilewati"); continue
        if not args.force and npz.exists() and rowf.exists():
            try:
                prev = json.loads(rowf.read_text(encoding="utf-8"))
                if prev.get("_cfg") == cfg and prev.get("status") == "OK":
                    prev.pop("_cfg"); rows.append(prev)
                    print(f"[{n}/{len(files)}] {p.name}: sudah ada, dilewati"); continue
            except Exception:
                pass
        try:
            # tracker video-mode menyimpan state antar-frame: reset tiap video
            hands.reset(); pose.reset()
            s = time.time()
            d, fps, w, h, decoded = process_video(p, hands, pose, args.every)
            meta = dict(file=p.name, label=label, fps=float(fps), width=w, height=h, config=cfg)
            tmp = npz.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, meta=json.dumps(meta), **d)
            tmp.replace(npz)
            row = quality_row(p.name, label, "OK", d, fps, w, h, decoded, time.time() - s)
            print(f"[{n}/{len(files)}] {p.name}: T={row['frames']} pose={row['pose_pct']}% "
                  f"tangan0/1/2={row['hand0_pct']}/{row['hand1_pct']}/{row['hand2_pct']}% {row['flags']}")
            rowf.write_text(json.dumps({**row, "_cfg": cfg}), encoding="utf-8")
        except Exception as e:
            row = quality_row(p.name, label, "FAILED", err=f"{type(e).__name__}: {e}")
            print(f"[{n}/{len(files)}] {p.name}: GAGAL {e}")
            traceback.print_exc(limit=1)
        rows.append(row)
    hands.close(); pose.close()

    cols = ["file", "label", "category", "status", "duplicate_of", "frames", "decoded_frames", "fps", "width", "height",
            "pose_pct", "hand0_pct", "hand1_pct", "hand2_pct", "left_pct", "right_pct",
            "longest_no_hand_gap", "flags", "proc_sec", "error"]
    with (out / "reports" / "video_quality.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["status"] == "OK"]
    summary = dict(config=cfg, total=len(rows), ok=len(ok),
                   failed=sum(r["status"] == "FAILED" for r in rows),
                   skipped_duplicate=sum(r["status"] == "SKIPPED_DUPLICATE" for r in rows),
                   flagged=sum(bool(r.get("flags")) for r in ok),
                   mean_pose_pct=round(float(np.mean([r["pose_pct"] for r in ok])), 2) if ok else None,
                   mean_hand0_pct=round(float(np.mean([r["hand0_pct"] for r in ok])), 2) if ok else None,
                   elapsed_sec=round(time.time() - t0, 1))
    (out / "reports" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRINGKASAN:", json.dumps(summary, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()

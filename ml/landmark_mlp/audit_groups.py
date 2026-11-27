"""Audit heuristik grup foto sebelum training; tidak membaca ulang gambar."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', default='landmarks.npz')
    ap.add_argument('--show', type=int, default=6, help='Jumlah contoh grup per kelas')
    a = ap.parse_args()
    if a.show < 1:
        ap.error('--show harus >= 1')
    with np.load(a.npz, allow_pickle=False) as z:
        if 'ver' not in z.files or int(z['ver']) != 3:
            raise SystemExit('NPZ bukan format v3. Jalankan extract.py versi terbaru.')
        files, cls, grp = z['files'], z['cls'], z['grp']
        class_names = [str(x) for x in z['class_names']]
    for c in class_names:
        by_group = defaultdict(list)
        for f, label, group in zip(files, cls, grp):
            if str(label) == c:
                by_group[str(group)].append(str(f))
        sizes = sorted(len(v) for v in by_group.values())
        if not sizes:
            print(f'\n{c}: TIDAK ADA DATA')
            continue
        median = sizes[len(sizes)//2]
        print(f'\n{c}: {len(files[cls == c])} foto, {len(sizes)} grup; ukuran grup min/median/max={sizes[0]}/{median}/{sizes[-1]}')
        keys = sorted(by_group)
        selected = keys[:a.show]
        if len(keys) > a.show:
            selected += ['...']
            selected += keys[-a.show:]
        for key in selected:
            if key == '...':
                print('  ...')
                continue
            items = sorted(by_group[key])
            print(f'  {key}: {len(items)} foto | awal={items[0]} | akhir={items[-1]}')
    print('\nCATATAN: grup dibentuk dari nomor file, bukan metadata sesi atau pemeriksaan kemiripan visual.')


if __name__ == '__main__':
    main()

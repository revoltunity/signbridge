# SignBridge — penerjemah Bahasa Indonesia ↔ SIBI (tahap pengembangan)

## Status web saat ini
- React + Vite; kamera diproses oleh MediaPipe di browser.
- Mesin pengenalan web menggunakan **MLP landmark statis**, bukan BiGRU 195 kelas.
- Model produksi `web/public/models/hand_mlp.json` dibuat otomatis oleh `ml/landmark_mlp/train.py`.
- Sebelum training selesai, pengenalan kamera belum dapat digunakan. Fixture sintetis di `web/tests/fixtures/` hanya untuk uji kesesuaian kode, bukan model kata sebenarnya.
- `ml/legacy_video_gru/` hanya arsip pipeline lama dan tidak dipanggil oleh web.

## Web (PowerShell)
```powershell
cd web
npm ci
npm test
npm run build
npm run dev
```
Buka alamat localhost yang ditampilkan Vite. Kamera memerlukan localhost atau HTTPS.

## Python (Windows / Python 3.11)
Dari folder utama proyek:
```powershell
py -3.11 -m venv C:\venvs\signbridge
C:\venvs\signbridge\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ml\requirements-landmark.txt scikit-learn
```

## Ekstraksi landmark dan training
```powershell
cd ml\landmark_mlp
python extract.py --data "C:\path\ke\IsyaratKata" --out landmarks.npz --block-size 30
python audit_groups.py --npz landmarks.npz
python train.py --npz landmarks.npz --out out
```

`--block-size 30` adalah **heuristik**: nomor foto dibagi per 30 foto di dalam setiap kelas, dihitung dari nomor terkecil. Ini bukan deteksi sesi yang sebenarnya. Jalankan `audit_groups.py` dan periksa apakah rentang nama foto tiap grup benar-benar satu sesi. Split train/validation/test memisahkan grup hasil heuristik ini, bukan otomatis menjamin bebas near-duplicate.

Ekstraksi dapat dilanjutkan (resume). Cache hanya digunakan bila path relatif, ukuran file, waktu modifikasi, dan grup cocok. Untuk memaksa ekstraksi ulang, hapus `landmarks.npz` terlebih dahulu.

Hasil training:
- `ml/landmark_mlp/out/model.json`
- `ml/landmark_mlp/out/report.json`
- `web/public/models/hand_mlp.json` (disalin otomatis jika folder web tersedia)

## Batasan model
MLP ini mengenali **pose statis**. Isyarat SIBI yang maknanya bergantung pada gerakan/urutan waktu tidak dapat dijamin benar dari satu pose. Kelas semacam itu perlu dikumpulkan sebagai klip video dan dievaluasi dengan model temporal (mis. landmark sequence + GRU/TCN), bukan dipaksakan sebagai klasifikasi foto.

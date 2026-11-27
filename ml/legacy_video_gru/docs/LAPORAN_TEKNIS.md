# Laporan Teknis — SignBridge

Tanggal: 26 Nov 2026. Semua angka di bawah berasal dari perintah yang benar-benar dijalankan; yang tidak bisa dijalankan dinyatakan eksplisit.

## 1. Berkas yang diterima
SIGNBRIDGE.zip (aplikasi + checkpoint), signbridge_next_step.zip (skrip ekstraksi, 195 `.npz` landmark, `training_dataset.npz`, skrip training), DATASET_SIBI.zip (199 video WebM; 195 kelas setelah duplikat `A (1)` dll. dan 3 label karantina). **Tidak ada:** `.onnx`/`export_onnx.py`, validation set, rekaman webcam nyata, `mediapipe.solutions` & model `.task` di sandbox (tanpa internet).

## 2. Temuan audit aplikasi lama
1. Jendela geser 128 frame **berurutan** pada ±15 fps (throttle 66 md) → butuh ±8,5 dtk gerak; buffer dikosongkan tiap tangan hilang. Training memakai **klip utuh** 29,97 fps, berpadding nol di akhir. Struktur masukan tidak pernah cocok.
2. Dua modul MediaPipe duplikat dengan ambang berbeda (0,5 vs 0,3/0,1 milik training), WASM dari CDN `@latest`/versi, modul `signTranslator.js` tak terpakai, `App.jsx.backup`.
3. Klasifikasi di thread UI, array baru tiap frame, ambang 10% yang menampilkan kandidat sebagai hasil.
4. Status "Model siap" ketika hanya detektor yang siap.

## 3. Model
- State dict `.pt` (24 tensor) dibaca tanpa torch; bentuk sesuai spesifikasi (LayerNorm 258, GRU 2 lapis bidirectional hidden 128, head 256→192→195).
- `signbridge_gru_weights.bin` == tensor `.pt`: **0 dari 24 tensor berbeda** (`tools/check_weights_and_train_acc.py <training_dataset.npz>`, dijalankan dari akar proyek).
- Implementasi NumPy independen dari `train_signbridge.py` memberi **195/195** top-1 pada urutan training (keyakinan min 0,88, median 0,98). Ini bukti tidak langsung bahwa arsitektur/urutan gerbang benar; PyTorch sendiri tidak terpasang sehingga perbandingan langsung ke `torch` **tidak dilakukan**.
- JS vs NumPy: selisih logit maks **1,9e-6** (6 urutan; `tests/model.test.mjs`).
- ONNX: **tidak dibuat** (tak ada `onnx`/`torch` & tanpa internet; klaim tak bisa diverifikasi). Alternatif: GRU JavaScript murni di Worker, bobot identik. Kompromi: ±136 md/inferensi (Node), 4 inferensi per isyarat (hipotesis kecepatan).
- `class_names.json`: 195 kelas, urutan = `sorted()` kodepoint (`toko` terakhir), identik di tiga lokasi; dipakai tanpa diubah.

## 4. Kesesuaian preprocessing training vs browser
| Aspek | Training | Browser | Status |
|---|---|---|---|
| Tata letak | pose 33×[x,y,z,vis] 132 + kiri 63 + kanan 63 | sama (`featureLayout.js`) | cocok (uji unit) |
| Slot tangan | 2 tangan: urut x wrist; 1 tangan: vs titik tengah bahu 11/12; slot = sisi gambar | algoritma sama | cocok (uji unit) |
| Mirror | video **tidak** mirror (pose[11].x>pose[12].x di 195/195 video) | frame mentah tidak dimirror (CSS hanya tampilan) | cocok |
| Koordinat | x,y 0–1 terhadap frame, tanpa normalisasi tambahan | sama | cocok |
| Landmark hilang | 0 + mask (NaN→0) | 0 | cocok |
| Frame rate | semua frame, 29,97 fps | resample nearest-neighbour berdasar timestamp ke 29,97 fps | cocok secara logika; diuji sintetis |
| Panjang | >128: `linspace(...).round()`; ≤128: nol di akhir | `packSequence` | **identik bit-per-bit di 195/195** urutan |
| Rasio | 16:9 (182×1280×720, 13×1920×1080) | meminta 16:9; peringatan bila beda | parsial |
| Detektor | `mp.solutions` Hands(complexity 0, det 0,3, track 0,1) + Pose(complexity 1) | Tasks HandLandmarker/PoseLandmarker (full), ambang dipetakan | **TIDAK terverifikasi numerik** |

Alasan baris terakhir: `mediapipe.solutions` tidak ada di paket 0.10.33 sandbox, tidak ada model offline, tidak ada internet. Perbedaan model hand "lite" (training) vs "full" (Tasks) dan jalur Python vs WASM dapat menggeser koordinat; **ini risiko terbesar yang tersisa pada sisi ekstraksi**.

## 5. Statistik training yang menentukan desain
Klip 45–228 frame (median 125; 1,5–7,6 dtk, median 4,2 dtk); 87 klip >128 (dikompres 1,0–1,78×), 108 dipadding. Jeda tanpa tangan awal median 0,30 dtk, akhir 0,40 dtk; celah internal 95% ≤0,38 dtk. Rentang tangan terpendek 1,13 dtk. Karena itu: pre-roll 0,5 dtk, isyarat selesai saat tangan hilang 0,5 dtk, abaikan <0,5 dtk, batas 8 dtk, dan setelah penahanan panjang harus turunkan tangan dulu (tidak ada output berulang).

## 6. Temuan penting: model sangat sensitif pada batas dan kecepatan
Uji pada landmark training (bukan data baru):
- Klip utuh asli → 195/195. Dipotong pada tangan pertama/terakhir ±0 frame → 28/195; ±9 → 113/195; ±18 → 170/195. Model memakai panjang/pose jeda awal–akhir sebagai petunjuk (hasil menghafal satu klip per kelas).
- Replay lewat mesin lengkap dengan frame diam sintetis (98 klip, tiap 2): top-1 **35/98** (dikonfirmasi 23, benar 19); kamera 15 fps 31/98; isyarat 1,5× lebih cepat 27/98 (tanpa hipotesis kecepatan: 10/98 pada 1,5×, 21/98 pada 1,25×).
- Segmen negatif (isyarat terpotong/kibasan, 109): dikonfirmasi keliru 11 pada ambang "Seimbang".
- Preset ambang dari tabel presisi (replay + negatif): Ketat 0,80/0,50, Seimbang 0,70/0,40, Longgar 0,50/0,25. **Bukan** kalibrasi dari data validasi nyata.

Implikasi: kecocokan 195/195 pada data training tidak berarti pengenalan nyata. Pada orang, kamera, dan tempo baru hasil diperkirakan jauh lebih buruk; ini tidak diukur. "Kecepatan alami" **tidak terbukti**: hipotesis kecepatan (1/1,25/1,5/2×) hanya memperbaiki replay sintetis.

## 7. Arsitektur realtime
`requestVideoFrameCallback` (cadangan rAF) → `extractFrame` → `engine.push({t, features, hasHand})` → segmenter → saat selesai: resample 29,97 fps → 4 hipotesis kecepatan → Worker (satu klasifikasi sekaligus; yang bentrok ditandai "dropped") → rata-rata probabilitas → keputusan (confirmed/uncertain) → penekan duplikat (1 dtk). State UI diperbarui tiap ≥150 md; bingkai ditangani lewat ref. Cleanup: stop track kamera, batalkan callback, `close()` MediaPipe, `terminate()` Worker. Latensi ≈ 0,5 dtk (tangan turun) + ±0,6 dtk inferensi (4×136 md di Node; belum diukur di browser).

## 8. Pengujian
**Lulus (dijalankan):** `npm test` (assets, model vs NumPy, pipeline, worker — 4/4); `tools/verify_against_dataset.mjs` (195/195 identik bit; 195/195 top-1 pada data training); bundel esbuild `src/main.jsx` tanpa impor tak terselesaikan (2,0 MB).
**Gagal/tidak bisa dijalankan:** `vite build` — `node_modules` bawaan hanya berisi biner Windows (esbuild/rollup win32) dan tak ada internet untuk `npm install`; jalankan `npm install && npm run build` di mesin Anda. Webcam & MediaPipe nyata; ekstraksi landmark ulang; perbandingan dengan `torch`/ONNX; ukuran performa di browser; akurasi pada rekaman baru; uji kecepatan alami nyata.

## 9. Keterbatasan & langkah eksternal
1. Satu video per kelas ⇒ generalisasi tidak diketahui dan kemungkinan rendah. Langkah terbaik berikutnya: rekam beberapa penanda/kamera/tempo per kelas (dengan jalur browser yang sama), tahan sebagian untuk validasi, lalu latih ulang dengan augmentasi panjang jeda awal/akhir dan kecepatan (disetujui terlebih dahulu; tidak dilakukan di sini).
2. Verifikasi paritas MediaPipe Python vs Web memakai satu video: jalankan `extract_landmarks.py` dan pipeline web pada video yang sama, bandingkan 258 fitur.
3. Tidak ada kelas "tanpa isyarat"; hanya segmentasi tangan + ambang yang mencegah keluaran palsu.
4. Hanya isyarat tunggal; bukan terjemahan kalimat. Teks→isyarat tidak tersedia.
5. Label 3 kelas dikarantina (Belum, Saya, Akhiran-Wati) tidak ada di model.
6. Model `.task` MediaPipe diunduh dari internet; untuk offline simpan ke `public/` dan ubah URL di `featureExtractor.js`.

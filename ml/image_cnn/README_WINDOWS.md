# SignBridge Training Pipeline (Windows)

## 1. Struktur dataset

Setiap folder langsung di bawah root dataset adalah satu kelas. File gambar dibaca langsung dari folder kelas; dataset asli tidak dipindahkan, disalin, atau diubah.

```text
D:\SignBridge\dataset\
  A\*.jpg
  B\*.jpg
  C\*.jpg
```

Format yang didukung: JPG/JPEG, PNG, BMP, WEBP.

## 2. Instalasi

Buka PowerShell di folder pipeline:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch sesuai perangkat. Untuk GPU NVIDIA, pilih perintah Windows/Pip/CUDA yang cocok di https://pytorch.org/get-started/locally/ . Untuk CPU-only, instalasi `pip install -r requirements.txt` sudah dapat digunakan; jika instalasi torch gagal, pasang build CPU dari selector resmi PyTorch terlebih dahulu, lalu jalankan requirements.

```powershell
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 3. Jalankan tahapan

Ganti path dataset sesuai lokasi sebenarnya.

```powershell
python .\signbridge_ml.py audit --data "D:\SignBridge\dataset" --work ".\signbridge_work"
python .\signbridge_ml.py split --data "D:\SignBridge\dataset" --work ".\signbridge_work"
python .\signbridge_ml.py train --data "D:\SignBridge\dataset" --work ".\signbridge_work" --epochs 20 --batch-size 32
python .\signbridge_ml.py evaluate --data "D:\SignBridge\dataset" --work ".\signbridge_work"
```

Jika proses training terputus, ulangi dengan `--resume`:

```powershell
python .\signbridge_ml.py train --data "D:\SignBridge\dataset" --work ".\signbridge_work" --epochs 20 --batch-size 32 --resume
```

`--epochs` adalah total target epoch, bukan jumlah epoch tambahan. Turunkan `--batch-size` jika kehabisan VRAM/RAM. Gunakan `--workers 0` bila Windows mengalami masalah multiprocessing.

## 4. Output

`signbridge_work/` berisi `audit_report.json`, `manifest.csv`, `split_summary.json`, `class_mapping.json`, `preprocessing.json`, `checkpoint_last.pt`, `model_best.pt`, dan `evaluation_report.json`.

Split dikelompokkan berdasarkan SHA256 sehingga salinan byte-identik tidak tersebar lintas split. File identik yang ditemukan pada label berbeda dikecualikan. Hash tidak mengenali foto near-duplicate atau subjek/sesi yang sama; bila tersedia metadata subjek/sesi, lakukan group split berdasarkan metadata sebelum menganggap evaluasi bebas leakage.

## 5. Integrasi React

Jangan memuat `.pt` langsung di browser. Jalankan inferensi melalui backend Python (mis. FastAPI) atau ekspor model ke ONNX dan gunakan runtime yang sesuai. Kontrak respons yang disarankan:

```json
{"label":"A","confidence":0.97,"is_unknown":false}
```

Untuk inferensi, samakan preprocessing `RGB`, resize 224x224, tensor float, dan normalisasi mean `[0.485,0.456,0.406]` serta std `[0.229,0.224,0.225]`. `class_mapping.json` menjadi sumber pemetaan indeks ke label. Threshold unknown belum dikalibrasi otomatis oleh pipeline versi ini; jangan menetapkan nilai produksi sebelum kalibrasi pada validation set dan pengujian gambar unknown.


### Kompatibilitas PyTorch 2.1 (CPU)

Pipeline mendukung PyTorch 2.1.x dan torchvision 0.16.x. Pada CPU, AMP/autocast dan GradScaler dinonaktifkan sepenuhnya; training memakai backward dan optimizer standar. Pada CUDA, pipeline memakai `torch.cuda.amp.autocast` dan `torch.cuda.amp.GradScaler`, yang kompatibel dengan PyTorch 2.1. Jika PyTorch sudah terpasang, periksa versinya dengan `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`.

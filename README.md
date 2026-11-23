# IF3170_G06_PhilippsLeatherJacket

Proyek ini mengimplementasikan pipeline klasifikasi *machine learning* untuk Tugas Besar 2 IF3170 Inteligensi Artifisial.  
Proyek ini mendukung dua mode penggunaan:
1. **Notebook Mandiri (*Self-Contained*) di `notebooks/`**: Setiap notebook dapat dijalankan langsung dari atas sampai bawah hanya dengan membaca file dari folder `data/` tanpa bergantung pada file/modul di `src/`.
2. **Arsitektur Modular di `src/` & `main.ipynb`**: Pipeline modular yang bersih untuk integrasi dan perbandingan seluruh model klasifikasi sekaligus.

---

## 📁 Struktur Direktori

```text
IF3170_G06_PhilippsLeatherJacket/
├── data/                        # Dataset kompetisi (satu-satunya dependensi data)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── output/                      # Hasil export model (.pkl) dan file prediksi (.csv)
├── notebooks/                   # Notebook mandiri (self-contained, hanya butuh data/)
│   ├── knn.ipynb                # Notebook lengkap KNN (Preprocessing + Modeling + Submission)
│   ├── dtl_experiment.ipynb     # Notebook eksperimen DTL (C4.5)
│   └── naive_bayes_experiment.ipynb # Notebook eksperimen Naive Bayes
├── src/                         # Modul logika utama (terstruktur & modular)
│   ├── __init__.py
│   ├── preprocess/              # Pipeline preprocessing modular
│   │   ├── config.py            # Konfigurasi & konstanta preprocessing
│   │   ├── missing.py           # Tahap 1: Missing value handling
│   │   ├── cleaning.py          # Tahap 2: Duplicate & invalid data handling
│   │   ├── encoding.py          # Tahap 3: Categorical data encoding
│   │   ├── scaling.py           # Tahap 4: Standardization (StandardScaler)
│   │   ├── balancing.py         # Tahap 5: Imbalance handling (SMOTE)
│   │   ├── feature_selection.py # Tahap 6: Feature selection (SelectKBest)
│   │   ├── reduction.py         # Tahap 7: Dimensionality reduction (PCA)
│   │   ├── dtl_features.py      # Feature engineering khusus DTL
│   │   ├── nb_features.py       # Feature engineering khusus Naive Bayes
│   │   └── preprocess.py        # Orkestrator preprocessing utama
│   ├── models/                  # Implementasi model-model klasifikasi
│   │   ├── __init__.py          # Export seluruh model
│   │   ├── base.py              # Interface BaseModel (fit & predict)
│   │   ├── knn.py               # Model 1: KNN scratch (weights, distance metrics)
│   │   ├── dtl.py               # Model 2: Decision Tree Learning (C4.5 scratch)
│   │   └── naive_bayes.py       # Model 3: Naive Bayes scratch
│   └── evaluation/              # Utilitas validasi & submisi bersama
│       ├── __init__.py
│       ├── validator.py         # Split data, StratifiedKFold, evaluasi metrik
│       └── submission.py        # Simpan & verifikasi model .pkl, export & validasi file .csv
├── main.ipynb                   # Notebook utama: EDA, Preprocessing, Pembandingan Semua Model, Submisi Akhir
├── README.md                    # Dokumentasi repositori
└── .gitignore
```

---

## ⚙️ Setup Environment

Sebelum menjalankan kode atau notebook, pastikan virtual environment sudah aktif dan dependensi terpasang:

```powershell
# Buat virtual environment (jika belum ada)
python -m venv .venv

# Aktifkan virtual environment
# Di Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Di Command Prompt (cmd):
.\.venv\Scripts\activate.bat
# Di Linux / macOS:
source .venv/bin/activate

# Install dependensi yang dibutuhkan
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib ipykernel
```

> **Catatan VS Code:** Saat membuka notebook (.ipynb), klik **Select Kernel** di pojok kanan atas, lalu pilih environment Python dari `.venv` (`Python 3.12 (.venv)`).

---

## 🚀 Cara Menjalankan Kode

### 1. Menjalankan Notebook Mandiri (`notebooks/*.ipynb`)
Folder `notebooks/` berisi notebook yang **sepenuhnya independen (self-contained)**.  
Notebook di folder ini tidak mengimpor apa pun dari `src/` dan hanya membutuhkan dataset dari `data/`. Setiap anggota dapat meletakkan notebook modelnya di sini (misal `notebooks/knn.ipynb`, `notebooks/dtl_experiment.ipynb`, `notebooks/naive_bayes_experiment.ipynb`) dan menjalankannya secara utuh dari awal sampai akhir.
- Buka notebook (misal `notebooks/knn.ipynb`) di VS Code / Jupyter Notebook.
- Pastikan kernel yang dipilih adalah `.venv`.
- Klik **Run All** (seluruh proses: pemuatan data, EDA, preprocessing lengkap, training, evaluasi, hingga penyimpanan model & file submisi akan langsung berjalan).

---

### 2. Menjalankan Notebook Utama (`main.ipynb`)
Notebook ini digunakan untuk evaluasi menyeluruh dan laporan perbandingan performa seluruh model secara modular.
- Buka `main.ipynb` di VS Code / Jupyter Notebook.
- Pastikan kernel yang dipilih adalah `.venv`.
- Jalankan cell secara berurutan atau klik **Run All**.
- Model terbaik akan otomatis tersimpan di folder `output/<nama_model>.pkl` dan prediksinya di `output/submission_<nama_model>.csv`.

---

### 3. Menjalankan Skrip / Modul Python Secara Langsung (CLI)
Setiap modul di `src/` dilengkapi dengan runnable test / smoke test sederhana, sehingga bisa dites langsung lewat terminal:

#### a. Uji Pipeline Preprocessing:
```powershell
python src/preprocess/preprocess.py
```

#### b. Uji Model Tertentu (Standalone Smoke Test):
```powershell
python src/models/knn.py
```

---

## 🛠️ Panduan Menambahkan Model Baru (Untuk Anggota Tim)

Bagi anggota yang membawa model dari branch lain:
1. **Untuk Notebook Mandiri**:
   - Taruh notebook lengkap modelmu di dalam folder `notebooks/<nama_model>.ipynb` (pastikan hanya membaca dari `data/` dan dapat dijalankan secara mandiri).
2. **Untuk Integrasi ke `src/` & `main.ipynb`**:
   - Buat class model di `src/models/<nama_model>.py` dengan mewarisi `BaseModel` (`from src.models.base import BaseModel`).
   - Ekspor class dan konfigurasi di `src/models/__init__.py`.
   - Tambahkan pemanggilan model pada bagian *Model Comparison* di `main.ipynb` menggunakan helper `cross_validate_configs` dan `train_and_evaluate`.

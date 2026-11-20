# IF3170_G06_PhilippsLeatherJacket

Proyek ini mengimplementasikan pipeline klasifikasi *machine learning* secara modular, mulai dari pemrosesan data (preprocessing), implementasi model, validasi/evaluasi, hingga pembuatan file submisi.

---

## Struktur Direktori

```text
IF3170_G06_PhilippsLeatherJacket/
├── data/                        # Dataset kompetisi
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── output/                      # Hasil export model (.pkl) dan file prediksi (.csv)
├── notebooks/                   # Notebook eksperimen individual / per branch
│   └── knn_experiment.ipynb     # Contoh notebook eksplorasi model tertentu
├── src/                         # Modul logika utama (terstruktur & reusable)
│   ├── __init__.py
│   ├── preprocess/              # Pipeline preprocessing 7 tahap
│   │   ├── config.py            # Konfigurasi & konstanta preprocessing
│   │   ├── missing.py           # Tahap 1: Missing value handling
│   │   ├── cleaning.py          # Tahap 2: Duplicate & invalid data handling
│   │   ├── encoding.py          # Tahap 3: Categorical data encoding
│   │   ├── scaling.py           # Tahap 4: Standardization (StandardScaler)
│   │   ├── balancing.py         # Tahap 5: Imbalance handling (SMOTE)
│   │   ├── feature_selection.py # Tahap 6: Feature selection (SelectKBest)
│   │   ├── reduction.py         # Tahap 7: Dimensionality reduction (PCA)
│   │   └── preprocess.py        # Orkestrator preprocessing utama
│   ├── models/                  # Implementasi model-model klasifikasi
│   │   ├── __init__.py
│   │   ├── base.py              # Interface BaseModel (fit & predict)
│   │   └── knn.py               # Model 1: KNN scratch
│   └── evaluation/              # Utilitas validasi & submisi bersama
│       ├── __init__.py
│       ├── validator.py         # Split data (memperhatikan SMOTE), StratifiedKFold, evaluasi metrik
│       └── submission.py        # Simpan & verifikasi model .pkl, export & validasi file .csv
├── main.ipynb                   # Notebook utama: EDA, Preprocessing, Pembandingan 5 Model, Submisi Akhir
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

> **Catatan VS Code:** Saat membuka notebook (.ipynb), klik **Select Kernel** di pojok kanan atas, lalu pilih environment Python dari `.venv` (`Python 3.12 (.venv tubes 2)`).

---

## Cara Menjalankan Kode

Proyek ini fleksibel dan mendukung beberapa cara eksekusi tergantung kebutuhan:

### 1. Menjalankan Notebook Utama (`main.ipynb`)
Notebook ini digunakan untuk laporan akhir dan evaluasi menyeluruh (EDA → Preprocessing → Membandingkan performa 5 model → Menghasilkan file submisi terbaik).
- Buka `main.ipynb` di VS Code / Jupyter Notebook.
- Pastikan kernel yang dipilih adalah `.venv`.
- Jalankan cell secara berurutan atau klik **Run All**.
- Model terbaik akan otomatis tersimpan di folder `output/<nama_model>.pkl` dan prediksinya di `output/submission_<nama_model>.csv`.

---

### 2. Menjalankan Notebook Eksperimen Tertentu (`notebooks/*.ipynb`)
Setiap anggota yang mengembangkan atau mentuning model tertentu di branch masing-masing dapat bekerja secara terisolasi di folder `notebooks/` (misalnya `notebooks/knn_experiment.ipynb`) tanpa mengubah `main.ipynb`.
- Buka notebook eksperimen yang diinginkan di dalam folder `notebooks/`.
- Jalankan cell untuk melihat hasil tuning hyperparameter atau confusion matrix khusus untuk model tersebut.

---

### 3. Menjalankan Skrip / Modul Python Secara Langsung (CLI)
Setiap modul di `src/` dilengkapi dengan runnable test / smoke test sederhana, sehingga bisa dites langsung lewat terminal tanpa membuka notebook:

#### a. Uji Pipeline Preprocessing:
Menjalankan pemrosesan data train dan test, lalu menampilkan dimensi data hasil PCA:
```powershell
python src/preprocess/preprocess.py
```

#### b. Uji Model Tertentu (Standalone Smoke Test):
Menjalankan training dan prediksi sampel untuk model tertentu:
```powershell
python src/models/knn.py
```

---

## 🛠️ Panduan Menambahkan Model Baru (Untuk Anggota Tim)

Bagi anggota yang membawa model dari branch lain:
1. **Buat file model di `src/models/<nama_model>.py`**:
   - Turunkan class dari `BaseModel` (`from src.models.base import BaseModel`).
   - Implementasikan method `fit(self, X, y)` (mengembalikan `self`) dan `predict(self, X)` (mengembalikan array prediksi).
   - Definisikan list konfigurasi default, misalnya `DEFAULT_<MODEL>_CONFIGS = [...]`.
2. **Ekspor di `src/models/__init__.py`**:
   - Tambahkan import class model dan konfigurasinya.
3. **Gunakan di Notebook**:
   - Untuk eksperimen pribadi, buat notebook baru di `notebooks/<model>_experiment.ipynb`.
   - Untuk notebook utama (`main.ipynb`), cukup tambahkan blok evaluasi model pada bagian *Model Comparison* menggunakan helper `cross_validate_configs` dan `train_and_evaluate`.

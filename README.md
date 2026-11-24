# IF3170 Tugas Besar 2 — Prediksi Pembatalan Reservasi Penginapan (Supervised Learning)

Repositori kelompok **G06 — PhilippsLeatherJacket** untuk Tugas Besar 2 IF3170 Inteligensi Artifisial Semester I 2026/2027.

## Deskripsi Singkat

Program ini menyelesaikan permasalahan prediksi pembatalan reservasi penginapan (`is_canceled`) menggunakan lima algoritma *supervised learning* berbasis NumPy, mencakup tahapan prapemrosesan data (*cleaning*, *encoding*, *scaling*, penanganan ketidakseimbangan kelas dengan SMOTE, seleksi fitur, dan reduksi dimensi PCA), evaluasi dan validasi terhadap `scikit-learn`, penyimpanan serta pemuatan model (`.pkl`), hingga pembuatan berkas submisi Kaggle (`.csv`).

Algoritma pembelajaran mesin yang diimplementasikan:

- **k-Nearest Neighbors (KNN)**: jarak Minkowski/Euclidean/Manhattan/Hamming berbasis *batch* serta pembobotan *uniform* dan *inverted distance*.
- **Decision Tree Learning (DTL)**: algoritma **C4.5** (*gain ratio*, atribut numerik dan kategorikal, penanganan *missing values* proporsional, serta visualisasi *top-N* percabangan).
- **Logistic Regression**: *Maximum Likelihood Estimation* dengan optimasi *Stochastic Gradient Ascent* (SGA) dan *mini-batch Adam*, dilengkapi regularisasi L1, L2, dan Elastic Net.
- **Support Vector Machine (SVM)**: *soft-margin* SVM dengan solver *Sequential Minimal Optimization* (SMO) orde kedua, kernel *Linear*, *Polynomial*, dan *Gaussian* (RBF), serta visualisasi bidang pemisah 2D.
- **Bayesian Network (Naive Bayes)**: Naive Bayes kategorikal/Gaussian dengan *Laplace smoothing*, diskritisasi MDL, graf struktur relasi atribut-kelas, serta optimasi *Weighted Naive Bayes*.

### Struktur Repositori

```text
IF3170_G06_PhilippsLeatherJacket/
├── data/                        # Dataset kompetisi (train.csv, test.csv, sample_submission.csv)
├── docs/
│   └── laporan/                 # Dokumen laporan LaTeX dan PDF final
├── notebooks/                   # Notebook eksperimen per algoritma
│   ├── knn.ipynb
│   ├── dtl_experiment.ipynb
│   ├── logistic_regression_experiment.ipynb
│   ├── svm_experiment.ipynb
│   └── naive_bayes_experiment.ipynb
├── output/                      # Model tersimpan (.pkl), visualisasi, dan submisi (.csv)
├── scripts/
│   └── naive_bayes/             # Skrip pencarian fitur dan eksperimen Naive Bayes
├── src/
│   ├── preprocess/              # Pipeline prapemrosesan data dan rekayasa fitur
│   ├── models/                  # Kelas KNN, C45DecisionTree, LogisticRegression, NaiveBayes
│   ├── svm/                     # Paket SVM (kernels, SMO, model, fitur, visualisasi)
│   └── evaluation/              # Utilitas split validasi, cross-validation, save/load, dan submisi
├── tests/                       # Unit test SVM, rekayasa fitur, dan checkpoint
└── main.ipynb                   # Notebook utama integrasi dan perbandingan seluruh model
```

## Cara Setup & Run

**Prasyarat:** Python 3.10+ dan `pip`.

```bash
# 1. Buat dan aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Pasang dependensi
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib ipykernel
```

### 1. Menjalankan Notebook Utama (`main.ipynb`)

Buka `main.ipynb` di Jupyter / VS Code menggunakan kernel `.venv`, lalu jalankan seluruh sel secara berurutan (**Run All**) untuk mengeksekusi EDA, prapemrosesan, pelatihan dan evaluasi kelima model, penyimpanan/pemuatan model, serta reproduksi berkas submisi Kaggle terbaik.

### 2. Menjalankan Notebook Eksperimen Per Model (`notebooks/*.ipynb`)

Setiap berkas di `notebooks/` memuat eksperimen rinci masing-masing algoritma beserta perbandingan terhadap `scikit-learn`, penyimpanan model `.pkl`, visualisasi bonus, dan pembuatan submisi model terkait:

- `notebooks/knn.ipynb`
- `notebooks/dtl_experiment.ipynb`
- `notebooks/logistic_regression_experiment.ipynb`
- `notebooks/svm_experiment.ipynb`
- `notebooks/naive_bayes_experiment.ipynb`

### 3. Menjalankan Pengujian dan Modul lewat Terminal

```bash
# Uji pipeline preprocessing
python3 src/preprocess/preprocess.py

# Smoke test model individual
python3 src/models/knn.py
python3 src/models/logistic_regression.py

# Jalankan seluruh unit test
python3 -m unittest discover -s tests -v
```

## Pembagian Tugas

| Anggota | NIM | Kontribusi |
|---|---|---|
| Stevanus Agustaf Wongso | `13524020` | Logistic Regression (SGA, mini-batch Adam, L1/L2/Elastic Net), eksperimen & perbandingan `scikit-learn`, integrasi `main.ipynb`, laporan |
| Jonathan Kris Wicaksono | `13524023` | Decision Tree Learning (C4.5), penyiapan fitur DTL, visualisasi pohon `top_n`, eksperimen & perbandingan `scikit-learn`, laporan |
| Made Branenda Jordhy | `13524026` | Naive Bayes & Weighted Naive Bayes, diskritisasi MDL & seleksi fitur, visualisasi graf Naive Bayes, eksperimen & perbandingan `scikit-learn`, laporan |
| Ray Owen Martin | `13524033` | Pipeline *preprocessing* modular, KNN, utilitas evaluasi & validasi bersama, EDA, laporan |
| Philipp Hamara | `13524101` | SVM (kernel Linear, Polynomial, Gaussian/RBF & solver SMO), rekayasa fitur & *checkpoint* SVM, visualisasi margin 2D, unit test, laporan |

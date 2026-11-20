# IF3170_G06_PhilippsLeatherJacket

Repositori Tugas Besar 2 IF3170 Inteligensi Artifisial.

## Struktur Utama
- `src/preprocess/` — Pipeline preprocessing modular 7 tahap (`missing.py`, `cleaning.py`, `encoding.py`, `scaling.py`, `balancing.py`, `feature_selection.py`, `reduction.py`, orkestrator `preprocess.py`).
- `src/models/` — Implementasi modular algoritma klasifikasi (`BaseModel`, `KNN`, dan tempat menambahkan 4 model lain).
- `src/evaluation/` — Utilitas evaluasi (`StratifiedKFold`, train/val split pasca-SMOTE, confusion matrix, tabel perbandingan) dan pembuatan file submisi.
- `notebooks/` — Notebook eksplorasi/eksperimen per branch (opsional).
- `main.ipynb` — Notebook akhir untuk EDA, preprocessing, evaluasi & perbandingan 5 model, dan export submisi.

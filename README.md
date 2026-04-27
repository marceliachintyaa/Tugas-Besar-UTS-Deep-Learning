# Tugas Besar Deep Learning
## S2 Teknik Elektro | ABK6BAB3 | Universitas Telkom

---

## Anggota Kelompok

| Nama | NIM | Email |
|---|---|---|
| Marcelia Chintya Hartakaadi | 1101223073 | marceliachintya@student.telkomuniversity.ac.id |
| Natasha Fedora Barus | 1101223205 | natashafbarus@student.telkomuniversity.ac.id |

---

## Deskripsi Project

Repository ini berisi pipeline Machine Learning end-to-end untuk dua dataset sebagai bagian dari UTS mata kuliah Deep Learning for Electrical Engineering (ABK6BAB3).

---

## Struktur Repository

```
├── Codingan/
│   ├── UTS_Deep Learning_Klasifikasi.ipynb  
│   └── UTS_Deep Learning_Regresi.ipynb       
│
├── Laporan Analisis/
│   ├── UTS_Deep Learning_Klasifikasi.pdf  
│   └── UTS_Deep Learning_Regresi.pdf       
│
└── README.md
```

---

## Dataset 1 — Klasifikasi Penyakit Kardiovaskular

**Konteks:** Membangun model klasifikasi untuk membantu Dr. Siti Rahmawati memprediksi risiko penyakit kardiovaskular pada pasien klinik dengan sumber daya terbatas.

- **Sumber:** [Cardiovascular Disease Dataset — Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Sampel:** 70.000 data pasien (68.739 setelah cleaning)
- **Task:** Binary Classification — CVD: ya (1) / tidak (0)

### Tahapan Pipeline

| Tahap | Keterangan |
|---|---|
| **01 Feature Engineering** | EDA distribusi/missing/outlier, medical validity cleaning, pembuatan fitur BMI/pulse pressure/MAP, heatmap korelasi, StandardScaler (fit pada train only, no leakage), feature selection korelasi >0.85 |
| **02 Model Baseline** | LR, KNN, DT, RF, SVM — split 80:20 & 70:30, CV k=5, evaluasi AUC/F1/Accuracy/Precision/Recall |
| **03 Hyperparameter Tuning** | GridSearchCV exhaustive RF (216 kombinasi, 1.080 fits per split) |
| **Bonus** | Threshold sweep analysis (0.30–0.70), inference simulation 10 pasien |

### Hasil Utama

| Keterangan | Nilai |
|---|---|
| Model terpilih | Random Forest (recall tertinggi: 0.6900) |
| AUC sebelum tuning | 0.7727 |
| AUC setelah tuning | **0.8057** (split 80:20) |
| Best Parameters | max_depth=10, n_estimators=300, min_samples_leaf=4, min_samples_split=10, max_features=sqrt |
| Threshold klinis | 0.40 → F1 = 0.7408, FN turun 30.5% (666 pasien lebih terdeteksi) |

---

## Dataset 2 — Regresi Prediksi Konsumsi Energi Bangunan

**Konteks:** Membangun model regresi untuk membantu Pak Budi (PT. Cahaya Nusantara Energy) memprediksi konsumsi daya listrik bangunan setiap 10 menit berdasarkan data sensor IoT.

- **Sumber:** [Appliances Energy Prediction — UCI ML Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Sampel:** ~19.000 interval 10 menit
- **Task:** Regression — prediksi konsumsi daya (Wh)

### Tahapan Pipeline

| Tahap | Keterangan |
|---|---|
| **01 Feature Engineering** | Log-transform target, temporal features, cyclical encoding sin/cos, lag features (1–144 step), rolling statistics, StandardScaler no-leakage |
| **02 Model Baseline** | Linear Regression & SVR (RBF) — split 80:20 & 70:30, CV k=5, evaluasi R²/MAE/RMSE/MAPE |
| **03 Hyperparameter Tuning** | Ridge → GridSearchCV, SVR → RandomizedSearchCV |
| **Bonus** | Error analysis per segmen konsumsi, inference simulation |

### Hasil Utama

| Keterangan | Nilai |
|---|---|
| R² sebelum feature engineering | 0.1694 |
| R² setelah feature engineering | **0.5739** (+238.9%) |
| Model terpilih | SVR RBF tuned |
| R²(log) setelah tuning | **0.7181** (split 80:20) |
| MAE setelah tuning | 28.03 Wh |
| Best Parameters SVR | C=0.5, γ=0.01, ε=0.2 |

---

## Cara Menjalankan

Kedua notebook dirancang untuk dijalankan di **Google Colab**.

### Requirements
```bash
pip install kagglehub scikit-learn pandas numpy matplotlib seaborn scipy
```

### Dataset 1
Dataset otomatis diunduh via KaggleHub saat notebook dijalankan:
```python
import kagglehub
path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
```

### Dataset 2
```python
import kagglehub
path = kagglehub.dataset_download("loveall/appliances-energy-prediction")
```

# Financial Transaction Fraud Detection & Customer Segmentation

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification%20%26%20Clustering-orange)](https://scikit-learn.org/)

Proyek Machine Learning end-to-end yang berfokus pada dua aspek utama analitik data keuangan: **Fraud Detection (Klasifikasi)** dan **Customer Segmentation (Clustering)**.

> *Proyek ini dikembangkan sebagai Final Capstone Project untuk kursus sertifikasi "Belajar Machine Learning untuk Pemula" oleh Dicoding Academy, dan telah ditingkatkan untuk memenuhi standar portofolio profesional.*

---

## Gambaran Umum & Problem Statement

Di sektor keuangan dan perbankan, transaksi penipuan merupakan risiko signifikan yang berpotensi menimbulkan kerugian finansial besar dan menurunkan kepercayaan nasabah. Di sisi lain, kurangnya pemahaman terhadap perilaku pelanggan menghambat institusi keuangan dalam menawarkan layanan yang dipersonalisasi.

Proyek ini menangani kedua tantangan tersebut melalui pipeline Machine Learning dua arah:

1. **Classification Pipeline** — Menggunakan supervised learning (`Decision Tree` dan `Random Forest`) untuk mengidentifikasi apakah suatu transaksi keuangan bersifat fraudulent atau legitimate.
2. **Clustering Pipeline** — Menggunakan unsupervised learning (`K-Means` + `PCA`) untuk mensegmentasi pelanggan berdasarkan demografi dan perilaku transaksi.

---

## Struktur Repository

```
├── data/
│   ├── data_clustering_inverse.csv     # Data hasil clustering (skala asli)
│   └── [historical_transaction_data]   # Dataset mentah dan yang telah dibersihkan
├── models/
│   └── tuning_classification.h5        # Model terbaik hasil tuning (serialized via joblib)
├── notebooks/
│   ├── [Klasifikasi]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
│   └── [Clumination]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
├── requirements.txt                    # Dependensi proyek dengan versi yang dipinned
└── README.md
```

---

## Alur Teknis & Implementasi

### 1. Classification (Supervised Learning)

- **Handling Imbalanced Data** — Data fraud secara alami sangat tidak seimbang. Keseimbangan kelas dilakukan menggunakan *downsampling* pada kelas mayoritas (`legitimate`) agar pelatihan model tidak bias.
- **Hyperparameter Tuning** — Mengimplementasikan `GridSearchCV` dan `RandomizedSearchCV` untuk mengoptimalkan kedalaman struktur dan kriteria pemisahan classifier.
- **Evaluation Metrics** — Model dievaluasi menggunakan *Accuracy, Precision, Recall,* dan *F1-Score* untuk memastikan sensitivitas tinggi dalam mendeteksi fraud sekaligus meminimalkan false positive.

### 2. Clustering (Unsupervised Learning)

- **Feature Engineering & Transformation** — Menggunakan `LabelEncoder` untuk fitur kategorikal dan `StandardScaler` untuk menangani perbedaan magnitudo antar fitur numerik.
- **Dimensionality Reduction** — Menggunakan *Principal Component Analysis* (`PCA`) untuk mengurangi dimensi sekaligus mempertahankan variansi maksimum, sehingga batas antar cluster lebih jelas.
- **Optimal Cluster Selection** — Menggunakan *Elbow Method* (via `Yellowbrick KElbowVisualizer`) dan *Silhouette Score* untuk menentukan jumlah segmen pelanggan ($k$) yang paling optimal secara matematis.

---

## Catatan Teknis & Self-Reflection

### Mengenai Performa Model yang Tinggi

Jika Anda meninjau metrik evaluasi akhir, model mencapai skor yang hampir sempurna (akurasi, presisi, dan recall tinggi). Dalam lingkungan produksi atau enterprise nyata, data keuangan sangat volatil dan kompleks sehingga skor seperti ini jarang ditemui.

Namun, untuk proyek ini:

- Data berasal dari **dataset kuratif edukatif** (Dicoding Academy), yang dirancang untuk menguji pemahaman dasar pipeline ML siswa, bukan untuk menghadapi entropi data dunia nyata dalam skala besar.
- Skor yang tinggi memvalidasi bahwa **pipeline feature engineering, alur downsampling, dan hyperparameter grid telah diimplementasikan dengan benar dan optimal** sesuai kerangka teori.
- Serialisasi model berhasil dilakukan menggunakan `joblib` (`tuning_classification.h5`), sehingga classifier siap untuk deployment atau integrasi API.

---

## Cara Menjalankan Secara Lokal

**1. Clone repository:**
```bash
git clone https://github.com/username_kamu/financial-fraud-detection-ml.git
cd financial-fraud-detection-ml
```

**2. Buat virtual environment:**
```bash
python -m venv .venv

# Aktivasi di Windows:
.venv\Scripts\activate

# Aktivasi di Mac/Linux:
source .venv/bin/activate
```

**3. Install dependensi:**
```bash
pip install -r requirements.txt
```

**4. Jalankan Notebook:**

Buka VS Code atau Jupyter Lab, navigasi ke direktori `notebooks/`, pilih `.venv` sebagai kernel aktif, lalu jalankan sel-selnya.

---

## Kredit & Penghargaan

| | |
|---|---|
| **Author** | Zahran Fardiaz |
| **Course Provider** | Dicoding Academy — *Belajar Machine Learning untuk Pemula* |
| **Project Date** | Mei 2026 |

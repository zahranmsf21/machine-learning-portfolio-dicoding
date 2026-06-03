# Financial Transaction Fraud Detection & Customer Segmentation

## Project Overview
This repository contains a comprehensive data science and machine learning project focused on **Financial Transaction Fraud Detection** (Classification) and **Customer Transaction Behavior Segmentation** (Clustering). 

Detecting fraudulent transactions is a critical challenge for modern banking and financial institutions. Failure to detect anomalies results in direct financial losses, while an overly aggressive system harms the user experience by blocking legitimate customers. This project provides a data-driven solution by deploying a robust machine learning pipeline to classify potential fraud and profile customer behaviors.

*Note: This project was developed as the Final Capstone Project for the **"Belajar Machine Learning untuk Pemula"** certification course by **Dicoding Academy**.*

---

## Project Structure
```text
├── data/
│   ├── data_clustering_inverse.csv     # Processed inverse data from clustering
│   └── [your_raw_data_files].csv       # Dataset utilized in the notebooks
├── models/
│   └── tuning_classification.h5        # Saved optimal Random Forest model (joblib format)
├── notebooks/
│   ├── [Klasifikasi]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
│   └── [Clustering]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
├── requirements.txt                    # Project dependencies and environment specs
└── README.md

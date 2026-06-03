# Financial Transaction Fraud Detection & Customer Segmentation

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification%20%26%20Clustering-orange)](https://scikit-learn.org/)

This repository contains a comprehensive End-to-End Machine Learning project focusing on two major aspects of financial data analytics: **Fraud Detection (Classification)** and **Customer Segmentation (Clustering)**.

*Note: This project was developed as the Final Capstone Project for the "Belajar Machine Learning untuk Pemula" certification course by Dicoding Academy. It has been enhanced and structured to meet professional portfolio standards.*

---

## Project Overview & Business Problem

In the financial and banking sector, fraudulent transactions present a significant risk, potentially leading to massive financial losses and diminished customer trust. Conversely, failing to understand customer behavior can prevent financial institutions from offering personalized services.

This project addresses both challenges using a dual-approach Machine Learning pipeline:
1. **Classification Pipeline:** Leveraging supervised learning (`Decision Tree` and `Random Forest`) to accurately identify whether a financial transaction is fraudulent or legitimate.
2. **Clustering Pipeline:** Utilizing unsupervised learning (`K-Means` coupled with `PCA`) to segment customers based on demographics and transaction behaviors for deeper business insights.

---

## Repository Structure

├── data/
│   ├── data_clustering_inverse.csv     # Exported clustered data with original scales
│   └── [historical_transaction_data]   # Raw and cleaned datasets
├── models/
│   └── tuning_classification.h5        # Serialized best-performing tuned model (joblib)
├── notebooks/
│   ├── [Klasifikasi]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
│   └── [Clumination]_Submission_Akhir_BMLP_Zahran_Fardiaz.ipynb
├── requirements.txt                    # Project dependencies with pinned versions
└── README.md

---

## Technical Workflow & Implementation

### 1. Classification (Supervised Learning)
* **Handling Imbalanced Data:** Financial fraud data is naturally heavily imbalanced. Class balancing was performed using systematic downsampling on the majority class (`legitimate`) to ensure unbiased model training.
* **Hyperparameter Tuning:** Implemented `GridSearchCV` and `RandomizedSearchCV` to optimize the structural depth and split criteria of the classifiers.
* **Evaluation Metrics:** Evaluated models using *Accuracy, Precision, Recall,* and *F1-Score* to ensure high sensitivity toward detecting fraud while minimizing false positives.

### 2. Clustering (Unsupervised Learning)
* **Feature Engineering & Transformation:** Applied `LabelEncoder` for categorical traits and `StandardScaler` to handle magnitude variances across numerical features.
* **Dimensionality Reduction:** Utilized Principal Component Analysis (`PCA`) to reduce dimensionality while preserving maximum variance, allowing cleaner boundary separations.
* **Optimal Cluster Selection:** Used the *Elbow Method* (via `Yellowbrick KElbowVisualizer`) and *Silhouette Scores* to mathematically determine the most optimal number of customer segments ($k$).

---

## Key Insights & Technical Notes (Self-Reflection)

### About the High Model Performance:
If you review the final evaluation metrics, the models achieve near-perfect metrics (high accuracy/precision/recall)]. In a production or real-world enterprise environment, financial data is highly volatile, noisy, and complex, making such near-perfect scores rare. 

However, for this specific project:
* The data originates from a curated education dataset (Dicoding Academy), designed to test a student's foundational understanding of ML pipelines rather than dealing with massive real-world data entropy.
* The near-perfect metrics validate that the **feature engineering pipeline, downsampling workflow, and hyperparameter grids were implemented correctly and optimally** according to theoretical frameworks.
* Model serialization was successfully done using `joblib` (`tuning_classification.h5`), making the classifier ready for potential deployment or API integration.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
   git clone [https://github.com/username_kamu/financial-fraud-detection-ml.git](https://github.com/username_kamu/financial-fraud-detection-ml.git)
   cd financial-fraud-detection-ml
```

2. **Set up a virtual environment:**
```bash
python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on Mac/Linux:
   source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Notebooks:**
Open VS Code or Jupyter Lab, navigate to the `notebooks/` directory, select your `.venv` as the active kernel, and execute the cells.

📜 **Credentials & Acknowledgement**
    * Author: Zahran Fardiaz
    * Course Provider: Dicoding Academy (Belajar Machine Learning untuk Pemula)
    * Project Date: May, 2026

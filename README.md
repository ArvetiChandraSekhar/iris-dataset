# 🌸 Iris Species Classification — ML Model Comparison

A hands-on machine learning project that trains and evaluates **7 classification models** on the classic Iris dataset. The focus is on understanding the full ML pipeline and interpreting model performance through **precision, recall, F1-score, and confusion matrices**.

---

## Project Overview

This notebook demonstrates:
- How to load, explore, and visualize a dataset
- How to split data and encode labels correctly (without data leakage)
- How to train multiple classification models with scikit-learn, XGBoost, and MLPClassifier
- How to evaluate models using classification metrics and confusion matrices
- How train/test split ratio affects model performance

> **Why Iris?** It's a small, clean, balanced dataset — ideal for comparing algorithms side by side. Setosa is linearly separable from the other two classes; Versicolor and Virginica are not — making it a great benchmark for both linear and non-linear classifiers.

---

## Project Structure

```
iris-classification/
│
├── iris-dataset.ipynb       # Main Jupyter notebook
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## Models Trained

| Model | Type | Kernel / Config |
|---|---|---|
| Logistic Regression | Linear | OvR (One-vs-Rest) |
| K-Nearest Neighbors | Instance-based | k = 3 |
| Support Vector Machine | Kernel-based | Linear |
| Support Vector Machine | Kernel-based | RBF |
| Support Vector Machine | Kernel-based | Polynomial |
| XGBoost | Ensemble (Gradient Boosting) | `mlogloss` eval metric |
| MLP Classifier | Neural Network | `max_iter=1000` |

---

## Evaluation Metrics

Each model is evaluated using:

- **Accuracy** — Overall fraction of correct predictions
- **Precision** — Of predicted positives, how many are truly positive (weighted avg)
- **Recall** — Of actual positives, how many were correctly found (weighted avg)
- **F1 Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of per-class predictions vs. ground truth

---

## 🔍 Key Findings

- With an **80/20 train-test split**, all 7 models achieve near-perfect accuracy — a reflection of how learnable the Iris patterns are with sufficient data.
- With a **10/90 train-test split** (only 15 training samples), accuracy drops significantly across all models, especially the MLP. This validates that high scores are not artefacts of overfitting, but genuinely data-dependent.
- **Petal length and petal width** are the strongest discriminators between species.
- The **SVM with RBF kernel** and **KNN** handle the Versicolor/Virginica overlap most robustly.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ArvetiChandraSekhar/iris-dataset.git
cd iris-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook iris-dataset.ipynb
```

> **Note:** The dataset is downloaded automatically via `kagglehub`. You'll need a Kaggle account and API credentials configured. See [Kaggle API setup](https://www.kaggle.com/docs/api).

---

## Requirements

```
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Or install all at once:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

---

## Concepts Covered

| Concept | Description |
|---|---|
| Train/Test Split | Partitioning data to evaluate generalization |
| Label Encoding | Converting string classes to integers without leakage |
| Linear Separability | Why some classifiers work better on this dataset |
| Weighted Averaging | How multi-class metrics account for class support |
| Confusion Matrix | Per-class breakdown of model predictions |
| Overfitting Detection | How split ratios expose data dependency |

---

## Sample Outputs

- Pairplot of all feature combinations colored by species
- Sepal Length vs. Sepal Width scatter plot
- Petal Length vs. Sepal Width scatter plot
- Correlation heatmap of all features
- Confusion matrix for each model
- Bar chart comparison of accuracy, precision, recall, F1 across all models


<img width="2138" height="1045" alt="image" src="https://github.com/user-attachments/assets/8a907283-b31f-469c-ba65-d46a9cd36603" />

<img width="2152" height="1062" alt="image" src="https://github.com/user-attachments/assets/25bfcbe5-cf38-459b-a635-28cf30eaf03d" />

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

Made by **Chandra Sekhar** · [GitHub](https://github.com/ArvetiChandraSekhar) · [LinkedIn](https://linkedin.com/in/chandra-sekhar-arveti/)

---

> *"The Iris dataset may be old, but the principles it teaches are timeless — clean pipelines, honest evaluation, and understanding your data before your models."*

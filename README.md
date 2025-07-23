# breast-cancer-risk-analyzer

A machine learning–powered web application for early detection of breast cancer using the **10 most important diagnostic features**.  
Built with **Streamlit** and deployed locally.

---

## Overview

This project aims to help identify the risk of breast cancer by analyzing a patient's diagnostic data.  
It uses a trained **Support Vector Classifier (SVC)** model for predictions, with features selected based on importance scores.

The app includes:
- Streamlit web UI with input sliders
- Sidebar with developer info and GitHub
- Prediction probability output
- Model trained only on top 10 features for efficiency
- Reset button for easy use

---

## Models Compared

Several models were trained and evaluated before selecting the final one:

| Model                 | Accuracy (approx.) |
|----------------------|--------------------|
| K-Nearest Neighbors  |  Compared |
| Random Forest        |  Compared |
| Logistic Regression  |  Compared |
| **SVC (Final Model)**|  Best overall |

---

## Files in This Repo

| File                    | Description                                           |
|-------------------------|-------------------------------------------------------|
| `app.py`                | Streamlit UI logic and API for local deployment       |
| `svc.pkl`               | Trained SVC model using top 10 features               |
| `scaler.pkl`            | StandardScaler used during model training             |
| `breastcancerml.ipynb`  | Full training + model comparison notebook             |
| `README.md`             | This file                                             |

---

## Notebook

[`breastcancerml.ipynb`](breastcancerml.ipynb)  
Contains:
- Data cleaning & preprocessing
- Feature selection via importance ranking
- Model training and evaluation
- Comparison of multiple ML models
- Accuracy, confusion matrices, insights

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/breast-cancer-risk-analyzer
   cd breast-cancer-risk-analyzer

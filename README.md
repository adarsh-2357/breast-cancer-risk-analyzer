# Breast Cancer Risk Analyzer

A machine learning–powered web application for early detection of breast cancer using the **10 most important diagnostic features**.  
Built using **Streamlit** and deployed on **Streamlit Cloud**.

---

## Live App

You can access and use the app directly via the link below:  
**[👉 Launch the App](https://breast-cancer-risk-analyzer-mjgmivkhzy9ghybcvnhd3a.streamlit.app/)**

---

## Project Overview

This project helps identify the risk of breast cancer by analyzing diagnostic data from patients.  
The core model is a trained **Support Vector Classifier (SVC)** using 10 top-ranked features selected for maximum predictive performance and efficiency.

### Key Features:
- Web-based UI developed in Streamlit
- Resettable input fields for easy usability
- Shows prediction probability along with classification
- Compares multiple ML models before selecting the best
- Deployed to Streamlit Cloud for online access

---

## Models Compared

Before selecting the final model, several machine learning algorithms were trained and evaluated on the same dataset:

| Model                 | Performance |
|----------------------|-------------|
| K-Nearest Neighbors  | Compared    |
| Random Forest        | Compared    |
| Logistic Regression  | Compared    |
| **Support Vector Classifier (Final)** | Best Overall |

---

## Repository Structure

| File                    | Description                                           |
|-------------------------|-------------------------------------------------------|
| `app.py`                | Streamlit web app code                                |
| `svc.pkl`               | Trained SVC model (on 10 features)                    |
| `scaler.pkl`            | StandardScaler object used for consistent preprocessing |
| `breastcancerml.ipynb`  | Jupyter Notebook with training, feature selection, and model comparisons |
| `README.md`             | Project documentation (this file)                    |

---

## Notebook: Model Development

See [`breastcancerml.ipynb`](breastcancerml.ipynb) for:
- Data cleaning and preprocessing steps
- Feature importance analysis and selection
- Training and tuning multiple classifiers
- Model evaluation with accuracy and confusion matrices
- Final model export (with `pickle`)

---

## Running the App Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-risk-analyzer
   cd breast-cancer-risk-analyzer
2. Install dependencies:
    ```bash
   pip install -r requirements.txt
3. Run the Streamlit App
   ```bash
   streamlit run app.py

---

## Credits
   Developed by Adarsh Kumar
   GitHub: adarsh-2357

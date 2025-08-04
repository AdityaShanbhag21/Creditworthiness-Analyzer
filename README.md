# 💳 AI-Powered Creditworthiness Analyzer

An intelligent machine learning web application to predict whether a loan applicant is creditworthy, built with Streamlit, scikit-learn, and pandas.

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

##  Overview

The **AI-Powered Creditworthiness Analyzer** predicts the probability of loan approval based on applicant information like income, employment status, credit score, and assets. This tool helps financial institutions or fintech platforms evaluate risk in a transparent and data-driven manner.

---

##  Live Demo

 *Coming Soon on Streamlit Sharing*

---

## Machine Learning Models used

-  Logistic Regression
-  Random Forest Classifier
-  XGBoost Classifier
---

## Features

-  Upload customer data CSV
-  Data preprocessing & cleaning
-  Train/test multiple ML models
-  Accuracy & evaluation metrics display
-  Interactive prediction for new customers
-  Streamlit-based user-friendly interface

---

##  Tech Stack

| Component        | Technology             |
|------------------|------------------------|
| Backend          | Python (pandas, sklearn, xgboost) |
| Web App UI       | Streamlit              |
| Model Training   | Logistic Regression, Random Forest, XGBoost |
| Deployment       | Streamlit Cloud |

---

##  Input Dataset Columns

```text
loan_id
no_of_dependents
education
self_employed
income_annum
loan_amount
loan_term
cibil_score
residential_assets_value
commercial_assets_value
luxury_assets_value
bank_asset_value
loan_status
````

---

## 🧪 How to Run Locally

###  Step 1: Clone the repo

```bash
git clone https://github.com/AdityaShanbhag21/AI-Powered-Creditworthiness-Analyzer-.git
cd AI-Powered-Creditworthiness-Analyzer-
```

###  Step 2: Create virtual environment (optional) & install dependencies

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

###  Step 3: Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Project Structure

```
AI-Powered-Creditworthiness-Analyzer-/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── trained models (.pkl)
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Future Enhancements

*  Real-time API integration with financial apps
*  Authentication for user dashboards
*  Time-series analysis of credit behavior
*  Deployment to Streamlit Sharing / HuggingFace

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

##  Dataset

You can use a variety of datasets from [Kaggle](https://www.kaggle.com/).   
I have used [Loan-Approval-Prediction-Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset).  

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Author

Built by **Aditya Shanbhag**. [GitHub](https://github.com/AdityaShanbhag21)  |

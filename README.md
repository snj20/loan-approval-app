# 💳 Loan Approval Predictor (Streamlit ML App)

An end-to-end Machine Learning application that predicts whether a loan application will be approved based on applicant details such as income, credit history, education, employment status, and property area.

This project covers the full ML lifecycle:
**EDA → Data Cleaning → Model Training → Deployment using Streamlit**

---

## 🚀 Live Demo
> https://snj20-loan-approval-app-app-pm5kjf.streamlit.app/ 

---

## 📌 Problem Statement

Financial institutions need to assess whether a loan applicant is likely to repay a loan.  
This project builds a predictive model to classify loan applications as:

- ✅ Approved  
- ❌ Not Approved  

based on demographic and financial attributes.

---

## 📊 Dataset

- Source: Kaggle Loan Prediction Dataset  
- Type: Binary Classification  
- Size: ~600 records  

### Key Features:
- Gender  
- Married  
- Dependents  
- Education  
- Self Employed  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Term  
- Credit History  
- Property Area  

### Target:
- `Loan_Status` → (Y/N → 1/0)

---

## 🔍 Exploratory Data Analysis (EDA)

Performed in `eda.ipynb`:

- Checked missing values
- Analyzed class distribution
- Visualized feature relationships
- Identified key predictors (especially **Credit History**)
- Observed skewness in income and loan distributions

---

## 🧹 Data Preprocessing

Instead of manual preprocessing, a **pipeline-based approach** was used:

### Missing Value Handling:
- Numerical → Median imputation  
- Categorical → Most frequent value  

### Encoding:
- One-hot encoding for categorical variables  

### Why Pipeline?
- Ensures consistency between training and deployment  
- Prevents data leakage  
- Makes the model production-ready  

---

## 🤖 Model

- Algorithm: **Random Forest Classifier**
- Framework: `scikit-learn`
- Pipeline: `ColumnTransformer + Pipeline`

### Steps:
1. Data split (Train/Test)
2. Preprocessing (imputation + encoding)
3. Model training
4. Evaluation

---

## 📈 Model Performance

- Accuracy: ~80–85% (depends on split)
- Evaluated using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix

---

## 🧠 Key Insight

> **Credit History is the strongest predictor of loan approval**, followed by income and loan amount.

---

## 🖥️ Streamlit Application

Built an interactive UI using Streamlit:

### Features:
- Clean two-panel layout
- Input form for applicant details
- Real-time prediction
- Approval probability display
- Dynamic result panel:
  
  - 🟢 Green → Approved  
  - 🔴 Red → Not Approved  

---



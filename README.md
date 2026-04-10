# 💳 Credit Card Default Prediction — ML Classification

A supervised machine learning project predicting whether a loan applicant is likely to default, using decision trees and ensemble methods on a real-world credit dataset.

---

## Objective

Build a robust classification model that accurately identifies high-risk loan applicants — directly relevant to credit risk management in retail banking and financial services.

---

## Why This Matters for Finance

Credit default prediction is one of the most commercially valuable applications of ML in banking. Accurate models reduce non-performing loan ratios, improve capital allocation, and inform lending policy. This project reflects my interest in applying data science to financial decision-making, aligned with my CFA studies.

---

## Repository Structure

- /notebook — Jupyter notebooks and Python scripts for data preprocessing, model training and evaluation
- README.md — Project overview and documentation

---

## Dataset Features

| Feature | Description |
|---|---|
| Checking account balance | Current account status |
| Credit history | Past repayment behaviour |
| Loan purpose | Reason for borrowing |
| Loan amount | Size of credit requested |
| Employment status | Job stability indicator |
| Personal demographics | Age, housing, dependents |

---

## Models Built

Baseline
- Decision Tree Classifier

Ensemble Methods
- Bagging (Bootstrap Aggregating)
- Random Forest
- Boosting (AdaBoost and Gradient Boosting)

Evaluation Metrics
- Accuracy
- Precision, Recall, F1
- ROC-AUC
- GINI Impurity for feature importance

---

## Key Findings

- Ensemble methods outperformed single decision trees significantly on recall for defaulters
- Credit history and checking account balance were the strongest predictors
- Class imbalance handling was critical — raw accuracy was misleading without it

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

---

## What I'd Do Next

- Add logistic regression as an interpretable baseline for regulatory compliance context
- Implement SHAP values for explainability — important in credit decisions under UAE CBUAE guidelines
- Build a simple scoring card output — the format actually used by credit analysts
- Test on UAE-specific credit bureau data if accessible

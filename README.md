# Heart Disease Prediction System

This project is a **Heart Disease Prediction System** built using machine learning techniques. The system leverages various classification algorithms to predict the likelihood of heart disease based on clinical patient data. The implementation is designed to provide insights into the efficacy of machine learning in medical diagnostics.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Scope](#future-scope)

---

## Introduction

Heart disease remains one of the most prominent causes of mortality globally. Early diagnosis and prediction are crucial for effective treatment and management. This project applies machine learning models to analyze clinical features such as age, cholesterol level, and blood pressure, providing a predictive system for detecting heart disease.

The implemented models include:
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression

---

## Features

- Data preprocessing with feature scaling, outlier detection, and encoding.
- Hyperparameter optimization using **GridSearchCV**.
- Evaluation metrics: Accuracy, Precision, Recall, and F1-score.
- Visualization of data distributions and model performances.
- Comparison of models based on their ability to predict heart disease.

---

## Dataset

The dataset used for this project is the **Heart Disease Dataset** sourced from [Kaggle](https://www.kaggle.com/johnsmith88/heart-disease-dataset). It contains clinical and demographic information of patients with attributes such as:
- `age`: Patient's age.
- `sex`: Gender of the patient.
- `chol`: Serum cholesterol in mg/dl.
- `trestbps`: Resting blood pressure.
- `thalach`: Maximum heart rate achieved.
- `target`: Output variable indicating the presence of heart disease (1 = yes, 0 = no).

---

## Technologies Used

- **Python**: Programming language.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Matplotlib, Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model implementation.
- **Kaggle API**: Dataset retrieval.

---

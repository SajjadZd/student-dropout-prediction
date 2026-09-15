
# Student Dropout Prediction

## Project Overview

This project uses Machine Learning to predict whether a student may be at risk of dropping out.

The project uses Logistic Regression as a binary classification model.

## Problem Statement

Student dropout can negatively affect students and educational institutions. The purpose of this project is to develop a predictive system that can identify potential dropout risk using student-related information.

## Dataset

The dataset contains 10,000 student records and 19 columns before preprocessing.

The dataset includes academic, demographic, socioeconomic, and enrollment-related information.

## Target Variable

The target variable is:

- 0 = Not Dropout
- 1 = Dropout

## Preprocessing

The following preprocessing steps were performed:

- Missing value analysis
- Missing value handling
- Duplicate checking
- Categorical value checking
- Removal of Student_ID
- One-Hot Encoding
- StandardScaler
- Train/Test Split

## Exploratory Data Analysis

EDA was performed using:

- Dropout distribution
- GPA analysis
- Attendance analysis
- Study hours analysis
- Assignment delay analysis
- Correlation analysis

## Model

Logistic Regression was used for binary classification.

Class imbalance was handled using class weighting during model training.

## Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Student Risk Prediction

The application accepts student information and generates:

- Dropout probability
- Risk level
- Predicted class

Risk levels used by the application:

- Below 40% = Low Risk
- 40% to below 70% = Medium Risk
- 70% or above = High Risk

## Educational Early-Warning Application

The project can serve as a prototype early-warning system. Educational staff could use predictions as an additional source of information when deciding whether a student may benefit from support such as counseling, tutoring, mentoring, or financial assistance.

The prediction should not be treated as a final judgment about a student.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio
- Joblib
- Google Colab
- GitHub

## Project Workflow

Dataset
→ Cleaning
→ EDA
→ Class Imbalance Handling
→ Train/Test Split
→ Preprocessing
→ Logistic Regression
→ Evaluation
→ Prediction
→ Deployment

## Author

Student Dropout Prediction — Machine Learning Internship Project

# Predicting the income of the US population using census data

## Introduction

This project aimed to investigate the factors contributing to the income of respondents to the US Census of 1994/95.

The target variable was a binary feature indicating whether the individual earned more or less than $50k.

## Exploratory data analysis

EDA techniques were used to understand the data and identify variables that could be dropped or engineered into more predictive features using binning techniques.

Features containing NA values were identified and an imputation strategy decided.

Numerical and categorical features were noted for later preprocessing.

## Preprocessing

A preprocessing pipeline was built using sklearn:
- Imputation using SimpleImputer
- Binning using custom transformers
- Scaling using StandardScaler
- Encodng Categorical variables using OHE
- Resampling using SMOTE

## Modelling

Three models were tested:

- Logistic Regression
- Random Forest
- GradientBoostedClassifer

Mlflow was used to track experiment performance.

## Results

Whilst the models showed promising ROC_AUC scores, this was likely due to the imabalance in the target class. The Precision Recall curve and F1 scores showed that none of the models performed particularly well.

The best model was a Gradient Boosted Classifier with no binning and resampling using SMOTE.

## Futher research 

The model selection was limited to underpowered models due to hardware and time limitations. Further research should test more advanced models including SVC, XGBoost and possible neural networks to better understand the performance ceiling.
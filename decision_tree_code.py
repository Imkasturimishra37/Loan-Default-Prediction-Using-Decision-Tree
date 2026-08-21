# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:02:44 2026

@author: ADMIN
"""

'''Business Problem:

A finance company provides loans to customers based on their financial background, income stability, credit history, and repayment capability.

The company wants to predict whether a customer is likely to default on a loan or repay successfully before approving the loan application.


Objective:

Build a Machine Learning Decision Tree Classification Model that predicts whether a customer will default (yes) or not default (no) using customer-related features.'''

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import joblib


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Data science practice\Machine_learning\All New DataSets\All New DataSets\Data sets\credit.csv")

# =========================================================
# DISPLAY DATA
# =========================================================

print(df.head())

# =========================================================
# INPUT AND OUTPUT SPLIT
# =========================================================

X = df.drop("default", axis=1)

y = df["default"]

# =========================================================
# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# =========================================================

categorical_columns = X.select_dtypes(include=['object']).columns

numerical_columns = X.select_dtypes(exclude=['object']).columns

# =========================================================
# PREPROCESSING
# =========================================================

# OneHotEncoder converts categorical text into numbers

preprocessor = ColumnTransformer(

    transformers=[

        (

            'cat',

            OneHotEncoder(handle_unknown='ignore'),

            categorical_columns

        )

    ],

    remainder='passthrough'

)

# =========================================================
# CREATE PIPELINE
# =========================================================

pipeline = Pipeline([

    ('preprocessing', preprocessor),

    (

        'model',

        DecisionTreeClassifier(

            criterion='entropy',
            max_depth=5,
            random_state=42

        )

    )

])

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# =========================================================
# TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# =========================================================
# SAVE PIPELINE
# =========================================================

save_path = r"C:\Users\ADMIN\Desktop\Data science practice\Machine_learning\Decision Tree/loan_default_pipeline.pkl"

joblib.dump(

    pipeline,
    save_path

)

print("Pipeline Saved Successfully")

# Loan-Default-Prediction-Using-Decision-Tree
Build a Machine Learning Decision Tree Classification Model to predict whether a customer will default (yes) or not default (no) using customer-related features.
Business Problem
A finance company provides loans to customers based on their financial background, income stability, credit history, and repayment capability. The company wants to predict whether a customer is likely to default on a loan or repay successfully before approving the loan application.
Machine Learning Workflow

Dataset → Feature/Target Split → Categorical Encoding → Train-Test Split → Decision Tree → Prediction → Evaluation → Model Saving

Preprocessing

Categorical features are identified automatically using Pandas.
Categorical variables are converted into numerical representations using OneHotEncoder.
handle_unknown='ignore' allows the model to handle unseen categories during prediction.
Numerical features are passed through without transformation.
Preprocessing and model training are combined into a single Scikit-learn Pipeline.

Model

Algorithm: Decision Tree Classifier
Criterion: Entropy
Maximum Depth: 5
Random State: 42

Model Evaluation

Accuracy
Confusion Matrix
Classification Report
Precision
Recall
F1-score

Model Deployment/Reuse
The complete preprocessing + trained Decision Tree pipeline is saved using joblib as:

loan_default_pipeline.pkl


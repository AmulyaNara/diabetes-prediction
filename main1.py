# diabetes_classification_pipeline.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------
# 1. Create folder for EDA plots
# ------------------------------
EDA_PLOTS_DIR = "eda_plots"
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

# ------------------------------
# 2. Load data
# ------------------------------
df = pd.read_csv("diabetes_dataset.csv")


# Drop any unnecessary index column if present (e.g., "Unnamed: 0")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ------------------------------
# 3. Create Outcome column & drop Fasting_Blood_Glucose
# ------------------------------
df["Outcome"] = (df["Fasting_Blood_Glucose"] > 125).astype(int)
df = df.drop(columns=["Fasting_Blood_Glucose"])

# ------------------------------
# 4. Imputation & Outlier Handling
# ------------------------------
# 4.1. Identify numeric vs. categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols.remove("Outcome") 
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# 4.2. Fill missing categorical values with mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# 4.3. For numeric columns: cap outliers using the IQR method, and fill any missing with median
for col in numeric_cols:
    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap values
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    # Fill missing numeric with median (if any)
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# ------------------------------
# 5. Basic pandas exploration
# ------------------------------
print("=== Dataset Shape ===")
print(df.shape, "\n")

print("=== Dataset Info ===")
print(df.info(), "\n")

print("=== Descriptive Statistics (Numerical) ===")
print(df.describe().T, "\n")

print("=== Null Values by Column ===")
print(df.isnull().sum(), "\n")

print("=== First 5 Rows ===")
print(df.head(), "\n")

# ------------------------------
# 6. Exploratory Data Analysis (EDA) & Saving Plots
# ------------------------------
# 6.1. Outcome distribution
outcome_counts = df["Outcome"].value_counts()
print("=== Outcome Distribution ===")
print(outcome_counts, "\n")

plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{EDA_PLOTS_DIR}/outcome_distribution.png")
plt.close()

# 6.2. Categorical features: value counts by Outcome & countplots
for col in categorical_cols:
    print(f"--- {col} Value Counts by Outcome ---")
    # Print count of each category partitioned by Outcome
    print(pd.crosstab(df[col], df["Outcome"]), "\n")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="Outcome", data=df)
    plt.title(f"{col} Counts by Outcome")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/countplot_{col}.png")
    plt.close()

# 6.3. Numeric features: histograms and scatter plots vs Outcome
for col in numeric_cols:
    # Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/histogram_{col}.png")
    plt.close()

    # Scatter plot: feature vs. Outcome (binary)
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df["Outcome"], alpha=0.3)
    plt.title(f"Scatter Plot of {col} vs. Outcome")
    plt.xlabel(col)
    plt.ylabel("Outcome (0 or 1)")
    plt.tight_layout()
    plt.savefig(f"{EDA_PLOTS_DIR}/scatter_{col}_vs_outcome.png")
    plt.close()

# 6.4. Correlation heatmap (numeric features + Outcome)
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols + ["Outcome"]].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
)
plt.title("Correlation Matrix (Numeric Features + Outcome)")
plt.tight_layout()
plt.savefig(f"{EDA_PLOTS_DIR}/correlation_heatmap.png")
plt.close()

# ------------------------------
# 7. Train-Test Split (75% train, 25% test), stratified on Outcome
# ------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print("=== After Train-Test Split ===")
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}\n")

# Save original training set for later use
X_train_original = X_train.copy()
y_train_original = y_train.copy()

# ------------------------------
# 8. Check for class imbalance in the training set
# ------------------------------
train_counts = y_train.value_counts()
print("=== Training Outcome Distribution ===")
print(train_counts, "\n")

imbalance_ratio = train_counts.min() / train_counts.max()
print(f"Imbalance ratio (min/max): {imbalance_ratio:.2f}")

apply_smote = imbalance_ratio < 0.5
print(f"Apply SMOTE? {'Yes' if apply_smote else 'No'}\n")

# ------------------------------
# 9. Apply SMOTE BEFORE preprocessing if needed
# ------------------------------
if apply_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("=== After SMOTE ===")
    print(f"Resampled training set shape: {X_train.shape}, {y_train.shape}\n")

# ------------------------------
# 10. Preprocessing pipelines
# ------------------------------
numeric_features = numeric_cols
categorical_features = categorical_cols

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    drop='first', sparse_output=False, handle_unknown='ignore'
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# Fit preprocessor on original training data (before SMOTE)
preprocessor.fit(X_train_original)

# Transform data
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
X_train_original_preprocessed = preprocessor.transform(X_train_original)

# ------------------------------
# 11. Define base models (no hyperparameters)
# ------------------------------
base_models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    # "SVC": SVC(probability=True, random_state=42),
    "KNeighbors": KNeighborsClassifier(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 12. Evaluate base models using cross-validation & holdout
base_results = []

for name, model in base_models.items():
    print(f"--- Evaluating Base Model: {name} ---")

    # 12.1. Cross-validation on training data (F1 score)
    cv_scores = cross_val_score(
        model, X_train_preprocessed, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # 12.2. Fit on entire training set
    model.fit(X_train_preprocessed, y_train)

    # 12.3. Metrics on original training set (unresampled)
    y_train_pred = model.predict(X_train_original_preprocessed)
    train_f1 = f1_score(y_train_original, y_train_pred)

    # 12.4. Metrics on test set
    y_test_pred = model.predict(X_test_preprocessed)
    test_f1 = f1_score(y_test, y_test_pred)

    # 12.5. Record metrics
    base_results.append(
        {
            "Model": name,
            "Train_F1": train_f1,
            "CV_Mean_F1": cv_mean,
            "CV_Std_F1": cv_std,
            "Test_F1": test_f1,
        }
    )

# Display base model performance
base_df = pd.DataFrame(base_results).set_index("Model")
print("=== Base Models: Train F1 vs. CV F1 vs. Test F1 ===")
print(base_df.round(4), "\n")

# 13. Determine which models to tune based on threshold
threshold = 0.05
to_tune = []
for idx, row in base_df.iterrows():
    if abs(row["Train_F1"] - row["CV_Mean_F1"]) > threshold:
        to_tune.append(idx)

print("Models flagged for tuning:", to_tune)

# ------------------------------
# 14. Hyperparameter grids
# ------------------------------
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
    },
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    },
    # "SVC": {
    #     "C": [0.1, 1, 10, 100],
    #     "kernel": ["linear", "rbf", "poly"],
    #     "gamma": ["scale", "auto"],
    # },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}

# 15. Perform hyperparameter tuning only for flagged models
tuned_estimators = {}
for name, base_model in base_models.items():
    if name in to_tune:
        print(f"--- Hyperparameter Tuning: {name} ---")
        param_dist = param_grids[name]
        rs = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            scoring="f1",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        rs.fit(X_train_preprocessed, y_train)
        tuned_estimators[name] = rs.best_estimator_
        print(f"Best parameters for {name}: {rs.best_params_}")
        print(f"Best CV F1 for {name}: {rs.best_score_:.4f}\n")
    else:
        # If the model isn't flagged for tuning, keep the base_model as-is
        tuned_estimators[name] = base_model
        print(f"--- Skipping hyperparameter tuning for: {name} ---\n")

# ------------------------------
# 16. Evaluate tuned (or skipped) models on train & test
# ------------------------------
final_results = []

for name, estimator in tuned_estimators.items():
    print(f"--- Evaluating Final Model: {name} ---")

    # 16.1. Train set metrics (on original training data)
    y_train_pred = estimator.predict(X_train_original_preprocessed)
    y_train_prob = estimator.predict_proba(X_train_original_preprocessed)[:, 1]

    train_metrics = {
        "Accuracy": accuracy_score(y_train_original, y_train_pred),
        "Precision": precision_score(y_train_original, y_train_pred),
        "Recall": recall_score(y_train_original, y_train_pred),
        "F1": f1_score(y_train_original, y_train_pred),
        "ROC_AUC": roc_auc_score(y_train_original, y_train_prob),
    }

    # 16.2. Test set metrics
    y_test_pred = estimator.predict(X_test_preprocessed)
    y_test_prob = estimator.predict_proba(X_test_preprocessed)[:, 1]

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1": f1_score(y_test, y_test_pred),
        "ROC_AUC": roc_auc_score(y_test, y_test_prob),
    }

    final_results.append(
        {
            "Model": name,
            "Train_Accuracy": train_metrics["Accuracy"],
            "Train_Precision": train_metrics["Precision"],
            "Train_Recall": train_metrics["Recall"],
            "Train_F1": train_metrics["F1"],
            "Train_ROC_AUC": train_metrics["ROC_AUC"],
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
        }
    )

# 17. Display comparison of final train vs. test metrics
final_df = pd.DataFrame(final_results).set_index("Model")
print("=== Final Model Performance: Train vs. Test ===")
print(final_df.round(4), "\n")

# 18. Identify best model (highest Test_F1) and show detailed report
best_model_name = final_df["Test_F1"].idxmax()
best_model = tuned_estimators[best_model_name]

print(f"*** Best Model on Test F1: {best_model_name} ***\n")
y_best_test_pred = best_model.predict(X_test_preprocessed)

print("=== Classification Report (Test) ===")
print(classification_report(y_test, y_best_test_pred), "\n")

cm = confusion_matrix(y_test, y_best_test_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title(f"{best_model_name} Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig(f"{EDA_PLOTS_DIR}/confusion_matrix_{best_model_name}.png")
plt.close()

# ------------------------------
# 19. Function to predict outcome + probability for new input
# ------------------------------
def predict_outcome(sample_dict):
    """
    Given a dictionary of raw feature values (matching the training columns),
    this function:
      1. Converts the dictionary into a single-row DataFrame.
      2. Applies the same preprocessing pipeline to those raw features.
      3. Uses the chosen 'best_model' to predict the binary Outcome (0 or 1).
      4. Returns both the predicted class label and the probability of class 1.
    """
    sample_df = pd.DataFrame([sample_dict])
    sample_preprocessed = preprocessor.transform(sample_df)
    pred = best_model.predict(sample_preprocessed)[0]
    prob = best_model.predict_proba(sample_preprocessed)[0][1]
    return pred, prob

# ------------------------------
# 20. Example usage with all columns
# ------------------------------
example_sample = {
    "Age": 56,
    "BMI": 37.3,
    "Waist_Circumference": 93.7,
    "HbA1c": 6.0,
    "Blood_Pressure_Systolic": 150.6,
    "Blood_Pressure_Diastolic": 76,
    "Cholesterol_Total": 119,
    "Cholesterol_HDL": 99.6,
    "Cholesterol_LDL":77.3,
    "GGT": 98.5,
    "Serum_Urate": 6.6,
    "Dietary_Intake_Calories": 169.8,
    "Family_History_of_Diabetes": 1,
    "Previous_Gestational_Diabetes": 1,
    "Sex": "Female",
    "Ethnicity": "Asian",
    "Physical_Activity_Level": "Moderate",
    "Alcohol_Consumption": "Moderate",
    "Smoking_Status": "Current",
}

prediction, probability = predict_outcome(example_sample)
print(f"Predicted Outcome: {prediction} (1 = Diabetic, 0 = Non-Diabetic)")
print(f"Probability of Diabetes: {probability:.4f}")



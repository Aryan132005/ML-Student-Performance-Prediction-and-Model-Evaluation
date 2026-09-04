import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Dataset

df = pd.read_csv("Day18_19_student_habits_performance.csv")

print("===== STUDENT PERFORMANCE ML PROJECT =====")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns.tolist())


# EDA

print("\n===== NUMERICAL COLUMNS =====")

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print(numerical_columns)

print("\n===== CATEGORICAL COLUMNS =====")

categorical_columns = df.select_dtypes(
    include="object"
).columns.tolist()

print(categorical_columns)


# Exam Score Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df["exam_score"], kde=True)
plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.show()


# Study Hours vs Exam Score

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="study_hours_per_day",
    y="exam_score"
)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours Per Day")
plt.ylabel("Exam Score")
plt.show()


# Attendance vs Exam Score

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="attendance_percentage",
    y="exam_score"
)
plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance Percentage")
plt.ylabel("Exam Score")
plt.show()


# Sleep Hours vs Exam Score

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="sleep_hours",
    y="exam_score"
)
plt.title("Sleep Hours vs Exam Score")
plt.xlabel("Sleep Hours")
plt.ylabel("Exam Score")
plt.show()


# Correlation Analysis

plt.figure(figsize=(12, 8))

correlation = df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# Correlation with Exam Score

print("\n===== CORRELATION WITH EXAM SCORE =====")

exam_correlation = correlation["exam_score"].sort_values(
    ascending=False
)

print(exam_correlation)


# Feature and Target

print("\n===== FEATURE AND TARGET =====")

X = df.drop(
    ["exam_score", "student_id"],
    axis=1
)

y = df["exam_score"]

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print("exam_score")


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Identify Numerical and Categorical Features

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# Linear Regression Model

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

linear_model.fit(
    X_train,
    y_train
)


# Regression Predictions

y_pred = linear_model.predict(X_test)

print("\n===== LINEAR REGRESSION RESULTS =====")

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# Actual vs Predicted

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))


# Regression Train vs Test

train_predictions = linear_model.predict(X_train)

train_r2 = r2_score(
    y_train,
    train_predictions
)

test_r2 = r2_score(
    y_test,
    y_pred
)

print("\n===== REGRESSION TRAIN VS TEST =====")

print("Training R2 Score:", round(train_r2, 4))
print("Testing R2 Score:", round(test_r2, 4))

if train_r2 - test_r2 > 0.10:
    print(
        "Conclusion: There may be signs of overfitting."
    )
elif train_r2 < 0.50 and test_r2 < 0.50:
    print(
        "Conclusion: The model may be underfitting."
    )
else:
    print(
        "Conclusion: Training and testing performance "
        "are reasonably close."
    )


# Classification Target

print("\n===== CLASSIFICATION PROBLEM =====")

df["pass_fail"] = np.where(
    df["exam_score"] >= 50,
    "Pass",
    "Fail"
)

print(
    df["pass_fail"].value_counts()
)

print(
    "\nPassing Threshold = 50 Marks"
)


# Classification Features and Target

X_classification = df.drop(
    [
        "exam_score",
        "pass_fail",
        "student_id"
    ],
    axis=1
)

y_classification = df["pass_fail"]


# Train Test Split for Classification

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_classification,
    y_classification,
    test_size=0.20,
    random_state=42,
    stratify=y_classification
)


# Classification Feature Types

numeric_features_c = X_classification.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features_c = X_classification.select_dtypes(
    include="object"
).columns.tolist()


# Classification Preprocessor

classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features_c
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features_c
        )
    ]
)


# Random Forest Classification Model

classification_model = Pipeline(
    steps=[
        (
            "preprocessor",
            classification_preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

classification_model.fit(
    X_train_c,
    y_train_c
)


# Classification Predictions

y_pred_class = classification_model.predict(
    X_test_c
)

print("\n===== CLASSIFICATION RESULTS =====")

accuracy = accuracy_score(
    y_test_c,
    y_pred_class
)

precision = precision_score(
    y_test_c,
    y_pred_class,
    pos_label="Pass"
)

recall = recall_score(
    y_test_c,
    y_pred_class,
    pos_label="Pass"
)

f1 = f1_score(
    y_test_c,
    y_pred_class,
    pos_label="Pass"
)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))


# Classification Report

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test_c,
        y_pred_class
    )
)


# Confusion Matrix

cm = confusion_matrix(
    y_test_c,
    y_pred_class,
    labels=["Fail", "Pass"]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Classification Train vs Test

train_class_predictions = classification_model.predict(
    X_train_c
)

train_accuracy = accuracy_score(
    y_train_c,
    train_class_predictions
)

test_accuracy = accuracy_score(
    y_test_c,
    y_pred_class
)

print("\n===== CLASSIFICATION TRAIN VS TEST =====")

print(
    "Training Accuracy:",
    round(train_accuracy, 4)
)

print(
    "Testing Accuracy:",
    round(test_accuracy, 4)
)

if train_accuracy - test_accuracy > 0.10:
    print(
        "Conclusion: There may be signs of overfitting."
    )
elif train_accuracy < 0.60 and test_accuracy < 0.60:
    print(
        "Conclusion: The model may be underfitting."
    )
else:
    print(
        "Conclusion: Training and testing performance "
        "are reasonably close."
    )


# Feature Importance

print("\n===== FEATURE IMPORTANCE =====")

model = classification_model.named_steps["model"]

feature_names = (
    classification_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(10)
)


# Top Feature Importance Plot

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# Final Insights

print("\n===== 5 MEANINGFUL INSIGHTS =====")

top_correlation_feature = exam_correlation.drop(
    "exam_score"
).abs().idxmax()

print(
    "1. The feature showing the strongest correlation "
    "with exam_score is:",
    top_correlation_feature
)

print(
    "2. Linear Regression achieved an R2 score of:",
    round(test_r2, 4),
    "on the testing dataset."
)

print(
    "3. The average prediction error measured by MAE is:",
    round(mae, 2),
    "marks."
)

print(
    "4. The classification model achieved a testing "
    "accuracy of:",
    round(test_accuracy * 100, 2),
    "%."
)

print(
    "5. The most important classification features can "
    "be identified from the feature importance analysis."
)

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")
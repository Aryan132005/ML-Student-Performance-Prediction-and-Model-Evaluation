# 🤖 ML Student Performance Prediction & Model Evaluation

## 📌 Description

This project performs a complete Machine Learning workflow on a Student Performance dataset using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

The project starts with Exploratory Data Analysis (EDA) to understand the dataset and identify important factors related to student exam performance.

After EDA, two Machine Learning problems are performed:

1. Regression - Predicting the student's exam score.
2. Classification - Predicting whether a student will Pass or Fail.

The project also evaluates model performance and checks for possible overfitting or underfitting.

## 🎯 Objectives

The main objectives of this project are:

- Explore the Student Performance dataset
- Understand numerical and categorical variables
- Check missing values and duplicate records
- Analyze distributions of important variables
- Study relationships between features and exam scores
- Perform correlation analysis
- Prepare data for Machine Learning
- Apply feature preprocessing
- Split data into training and testing sets
- Build a Linear Regression model
- Predict exam scores
- Evaluate regression performance
- Create a Pass/Fail classification problem
- Train a classification model
- Evaluate classification performance
- Generate a confusion matrix
- Compare training and testing performance
- Identify overfitting or underfitting
- Identify important features
- Generate meaningful insights

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Jupyter Notebook
- CSV

## 📂 Dataset

The dataset used in this project is:

```text
Day18_19_student_habits_performance.csv
```

The dataset contains information related to student habits, lifestyle, and academic performance.

Important variables include:

- study_hours_per_day
- attendance_percentage
- sleep_hours
- social_media_hours
- netflix_hours
- exercise_frequency
- mental_health_rating
- exam_score

Other suitable student-related variables are also used during the analysis.

## 📂 Project Structure

```text
ML-Student-Performance/
│
├── Day18_19_student_habits_performance.csv
├── Day18_19_Student_Performance_ML.ipynb
└── README.md
```

## 🔍 Exploratory Data Analysis

The first step of the project is Exploratory Data Analysis.

The dataset is inspected using:

```python
df.head()
df.shape
df.columns
df.dtypes
df.info()
df.describe()
```

Missing values are checked using:

```python
df.isnull().sum()
```

Duplicate records are checked using:

```python
df.duplicated().sum()
```

## 📊 Data Analysis

The project analyzes:

- Student study habits
- Attendance
- Sleep hours
- Social media usage
- Netflix usage
- Exercise frequency
- Mental health rating
- Exam scores

Different visualizations are used to understand the distribution and relationships in the dataset.

### Exam Score Distribution

A histogram is used to understand the distribution of exam scores.

### Study Hours vs Exam Score

A scatter plot is used to investigate whether study hours are related to exam performance.

### Attendance vs Exam Score

The relationship between attendance percentage and exam score is analyzed.

### Sleep Hours vs Exam Score

The relationship between sleep hours and exam performance is also investigated.

## 🔗 Correlation Analysis

A correlation matrix is created using numerical variables.

```python
correlation = df.select_dtypes(
    include=np.number
).corr()
```

A heatmap is used to visualize the correlations.

The correlation with `exam_score` is also calculated to identify variables that appear to have stronger relationships with exam performance.

## 🧹 Data Preprocessing

Before Machine Learning, the dataset is prepared using suitable preprocessing techniques.

The preprocessing includes:

- Selecting relevant features
- Separating features and target
- Identifying numerical columns
- Identifying categorical columns
- Scaling numerical variables
- Encoding categorical variables
- Splitting data into training and testing sets

The target variable for regression is:

```text
exam_score
```

## ✂️ Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The split is performed using:

```python
train_test_split()
```

A random state is used to make the results reproducible.

## 📈 Regression Problem

The first Machine Learning problem is regression.

The objective is to predict:

```text
exam_score
```

### Model Used

```text
Linear Regression
```

The model uses relevant student features such as:

- Study hours per day
- Attendance percentage
- Sleep hours
- Social media hours
- Netflix hours
- Exercise frequency
- Mental health rating
- Other suitable variables

## 📏 Regression Evaluation

The Linear Regression model is evaluated using:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = Mean Absolute Error
```

Lower MAE indicates better performance.

### Mean Squared Error (MSE)

MSE calculates the average squared prediction error.

Lower MSE indicates better performance.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

Lower RMSE indicates better performance.

### R² Score

R² measures how well the model explains the variation in the target variable.

A higher R² score generally indicates better model performance.

## 🎯 Classification Problem

The second Machine Learning problem converts the exam score into two classes.

The passing threshold is:

```text
50 Marks
```

The classification target is created as:

```text
Exam Score >= 50  → Pass
Exam Score < 50   → Fail
```

Therefore:

```text
Pass = 1
Fail = 0
```

The classification model uses the same prepared student dataset.

## 🌳 Classification Model

A Random Forest Classifier is used to predict whether a student will Pass or Fail.

```text
Random Forest Classifier
```

The model is trained using the training dataset and evaluated on the testing dataset.

## 📊 Classification Evaluation

The classification model is evaluated using:

### Accuracy

Accuracy represents the percentage of correctly classified students.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Precision measures how many students predicted as Pass actually passed.

### Recall

Recall measures how many actual passing students were correctly identified.

### F1-Score

F1-score is the harmonic mean of precision and recall.

It provides a balanced measure when both precision and recall are important.

## 🔲 Confusion Matrix

A confusion matrix is created to understand classification results.

It contains:

- True Positive
- True Negative
- False Positive
- False Negative

A heatmap is used to visualize the confusion matrix.

## ⭐ Feature Importance

Feature importance is calculated using the Random Forest model.

The most important features are identified based on their contribution to classification predictions.

A bar chart is created to display the top important features.

## ⚖️ Training vs Testing Performance

The training and testing performance of both models is compared.

This helps identify whether the models are:

### Overfitting

When the model performs very well on training data but significantly worse on testing data.

### Underfitting

When the model performs poorly on both training and testing data.

### Good Generalization

When training and testing performance are reasonably close.

## 📋 Model Evaluation Summary

The project evaluates:

| Model | Problem | Evaluation |
|---|---|---|
| Linear Regression | Exam Score Prediction | MAE, MSE, RMSE, R² |
| Random Forest Classifier | Pass/Fail Prediction | Accuracy, Precision, Recall, F1 |
| Random Forest Classifier | Pass/Fail Prediction | Confusion Matrix |

## 💡 Key Insights

The final analysis provides meaningful insights such as:

1. The feature with the strongest relationship with `exam_score` can be identified through correlation analysis.

2. Study habits and attendance can provide useful information about student academic performance.

3. Lifestyle factors such as sleep, social media usage, Netflix usage, and exercise can also be investigated for their relationship with exam scores.

4. Linear Regression provides a numerical prediction of student exam performance and can be evaluated using MAE, MSE, RMSE, and R².

5. The classification model predicts whether students are likely to pass or fail using the 50-mark threshold.

6. The confusion matrix helps identify correctly and incorrectly classified students.

7. Comparing training and testing performance helps determine whether the models generalize well or show signs of overfitting or underfitting.

## 🎓 Learning Outcomes

Through this project, the following Machine Learning concepts are demonstrated:

- Exploratory Data Analysis
- Data inspection
- Missing value analysis
- Numerical and categorical feature identification
- Data visualization
- Correlation analysis
- Feature selection
- Data preprocessing
- One-hot encoding
- Feature scaling
- Train-test split
- Regression
- Linear Regression
- Classification
- Random Forest
- Model prediction
- MAE
- MSE
- RMSE
- R² Score
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance
- Overfitting
- Underfitting
- Model comparison

## ▶️ How to Run

### Step 1: Open Google Colab

Open Google Colab and create a new notebook.

### Step 2: Upload the Dataset

Upload:

```text
Day18_19_student_habits_performance.csv
```

### Step 3: Upload the Notebook

Open:

```text
Day18_19_Student_Performance_ML.ipynb
```

### Step 4: Run All Cells

Run all notebook cells from top to bottom.

The notebook will perform:

```text
Data Loading
      ↓
EDA
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Linear Regression
      ↓
Regression Evaluation
      ↓
Pass/Fail Creation
      ↓
Classification
      ↓
Confusion Matrix
      ↓
Classification Evaluation
      ↓
Train vs Test Comparison
      ↓
Feature Importance
      ↓
Final Insights
```

## 📌 Conclusion

This project demonstrates a complete Machine Learning workflow for analyzing and predicting student performance.

The regression model predicts numerical exam scores, while the classification model predicts whether a student is likely to pass or fail based on a 50-mark threshold.

EDA and correlation analysis help identify important relationships between student habits and academic performance. Model evaluation metrics provide a clear understanding of prediction quality, while training and testing comparisons help identify possible overfitting or underfitting.

Overall, the project demonstrates how student lifestyle, study habits, attendance, and other relevant factors can be analyzed using Machine Learning techniques to understand academic performance.

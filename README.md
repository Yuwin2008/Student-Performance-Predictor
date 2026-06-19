#  Student Performance Prediction using Machine Learning

##  Overview

This project applies a complete Machine Learning workflow to predict student mathematics performance using demographic, educational, and academic attributes.

The notebook follows the methodology presented in *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* and demonstrates a real-world end-to-end machine learning pipeline.

The objective is to predict a student's **Math Score** while exploring the factors that influence academic performance.

---

##  Problem Statement

Can we accurately predict a student's mathematics score using information such as:

- Gender
- Race/Ethnicity
- Parental Education Level
- Lunch Type
- Test Preparation Status
- Reading Score
- Writing Score

The project explores this question using regression models and rigorous evaluation techniques.

---

##  Dataset

The dataset contains information about student demographics and examination performance.

### Features

| Feature | Description |
|----------|-------------|
| gender | Student gender |
| race/ethnicity | Student ethnicity group |
| parental level of education | Parent education level |
| lunch | Lunch type |
| test preparation course | Test preparation completion status |
| math score | Mathematics score (Target) |
| reading score | Reading score |
| writing score | Writing score |

### Dataset Size

- 1000 Students
- 8 Original Features

---

##  Exploratory Data Analysis

The first stage focused on understanding the dataset.

### Dataset Inspection

Performed:

- Data Structure Analysis
- Summary Statistics
- Data Type Inspection
- Missing Value Detection

### Findings

- Dataset contains no missing values.
- Student scores are approximately normally distributed.
- Reading and writing scores are strongly related.

---

## Correlation Analysis

A correlation matrix was generated to investigate relationships among numerical variables.

### Key Insights

- Reading Score and Writing Score have a very strong positive correlation.
- Math Score is strongly correlated with Reading and Writing Scores.
- Academic performance tends to be consistent across subjects.

---

##  Data Visualization

Several visualizations were used to explore the data.

### Histograms

Used to analyze score distributions and identify trends.

### Scatter Plots

Explored relationships between:

- Reading Score
- Writing Score
- Math Score

### Enhanced Scatter Plot

Visualized:

- Bubble Size → Total Student Performance
- Color → Mathematics Performance

This allowed multiple dimensions of performance to be analyzed simultaneously.

---

##  Feature Engineering

Additional features were created to support analysis.

### Total Score

```python
total_score = math_score + reading_score + writing_score
```

Used to visualize overall student achievement.

### Score Categories

Students were grouped into score ranges to support stratified sampling and analysis.

> Note: Engineered features were primarily used for exploration and visualization and were excluded from final model evaluation when appropriate to prevent target leakage.

---

##  Data Preprocessing Pipeline

A complete Scikit-Learn preprocessing pipeline was built.

### Numerical Features

#### Median Imputation

```python
SimpleImputer(strategy="median")
```

Included to ensure robustness and production readiness.

#### Feature Scaling

```python
StandardScaler()
```

Standardized numerical variables.

### Categorical Features

#### One-Hot Encoding

```python
OneHotEncoder()
```

Converted categorical variables into machine-readable numerical representations.

Example:

```text
Male   → [1,0]
Female → [0,1]
```

### Column Transformer

Both pipelines were combined using:

```python
ColumnTransformer()
```

This ensures consistent preprocessing during training and inference.

---

##  Model Training

The target variable selected for prediction was:

```text
Math Score
```

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

---

##  Linear Regression

A Linear Regression model was used as the baseline model.

### Test Performance

```text
RMSE ≈ 5.36
```

Interpretation:

> On average, predictions differ from actual math scores by approximately 5.36 points.

---

##  Cross Validation

To obtain a more reliable estimate of model performance, 5-Fold Cross Validation was performed.

### RMSE Scores

```text
5.29
5.69
5.14
4.99
5.42
```

### Mean RMSE

```text
5.31
```

### Conclusion

The model demonstrated stable performance across multiple validation folds and showed good generalization ability.

---

##  Random Forest Regressor

A Random Forest Regressor was implemented to investigate whether a more complex nonlinear model could outperform Linear Regression.

---

##  Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model.

### Parameters Tuned

- n_estimators
- max_features
- bootstrap

Multiple parameter combinations were evaluated using 5-fold cross-validation.

---

##  Model Comparison

| Model | Cross Validation RMSE | Test RMSE |
|---------|---------:|---------:|
| Linear Regression | **5.31** | **5.36** |
| Random Forest (Tuned) | 6.06 | 6.02 |

---

##  Final Results

The Linear Regression model achieved the best overall performance.

### Best Model

```text
Linear Regression
```

### Best Cross Validation RMSE

```text
5.31
```

### Best Test RMSE

```text
5.36
```

### Key Finding

Despite evaluating and tuning a Random Forest model, Linear Regression produced superior results.

This suggests that the relationships between the predictor variables and mathematics performance are largely linear.

---

##  Machine Learning Concepts Demonstrated

This project covers:

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Data Visualization
- Feature Engineering
- Stratified Sampling
- Data Preprocessing Pipelines
- Missing Value Handling
- Feature Scaling
- One-Hot Encoding
- Linear Regression
- Random Forest Regression
- Cross Validation
- Grid Search Hyperparameter Tuning
- Model Evaluation using RMSE
- Model Persistence using Joblib

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebook

---

##  Future Improvements

Potential enhancements include:

- XGBoost Regressor
- Feature Importance Analysis
- SHAP Explainability
- Streamlit Web Application
- Model Deployment
- Automated Training Pipeline

---

##  Author

Developed as part of a machine learning learning journey following the workflow presented in *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

---

 If you found this project interesting, consider giving the repository a star.

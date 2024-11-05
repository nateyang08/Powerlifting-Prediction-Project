# Powerlifting Prediction Data Science Project README

## Overview
This repository contains Python code developed for analyzing and modeling powerlifting data sourced from the OpenPowerlifting dataset. The project focuses on evaluating different predictive models for estimating the performance of powerlifters based on their personal characteristics and lift records. It employs various data preprocessing techniques, exploratory data analysis (EDA), regression models, and Principal Component Analysis (PCA) for feature reduction.

The primary goal of this project is to predict powerlifting metrics such as the best squat, bench press, and deadlift weights using attributes like age, body weight, and lift type. This project aims to provide a comprehensive analysis that can assist coaches, athletes, and researchers in understanding the factors influencing powerlifting performance.

## Features
### 1. **Data Loading and Preprocessing**
   - Loads the OpenPowerlifting dataset from a CSV file.
   - Filters out rows with missing values and unwanted entries (e.g., disqualified lifters).
   - Encodes categorical variables numerically:
     - Sex: 'M' = 1.0, 'F' = 2.0, 'Mx' = 3.0.
     - Equipment: 'Raw' = 1.0 (only raw equipment lifters are considered).
   - Converts relevant columns to numeric data types.
   - Further filters the dataset to include only raw male lifters for a targeted analysis.

### 2. **Exploratory Data Analysis (EDA)**
   - **Correlation Matrix**: Visualizes relationships between age, body weight, and the best squat, bench, and deadlift performances using a heatmap.
   - **Variance Inflation Factor (VIF)**:
     - Calculates VIF for predictor variables to assess multicollinearity, ensuring that no feature is highly collinear with another.

### 3. **Regression Models**
   The project employs multiple regression techniques to predict each lift metric (squat, bench, and deadlift). Performance is evaluated using metrics such as Mean Absolute Error (MAE), R² (R-squared), and tolerance-based accuracy (predictions within 10% of actual values).
   
   - **Linear Regression**:
     - Basic linear model as a baseline.
     - Evaluation metrics include MAE, R², and tolerance accuracy.
   - **Ridge Regression**:
     - A regularized linear model to manage multicollinearity and improve generalization.
     - Evaluated using the same metrics as the linear model.
   - **XGBoost Regressor**:
     - Gradient boosting technique for robust, non-linear modeling.
     - Utilizes hyperparameters like `n_estimators`, `learning_rate`, and `max_depth`.
   - **RandomForest Regressor**:
     - Ensemble-based model that provides a balance between bias and variance.
     - Trains and evaluates using similar metrics as other models.

### 4. **Principal Component Analysis (PCA)**
   - **Feature Scaling**: Standardizes predictor variables before applying PCA.
   - **Dimensionality Reduction**:
     - Reduces data dimensionality to three principal components.
     - Provides explained variance ratios and cumulative variance to assess component importance.
   - **Scree Plot**:
     - Visualizes variance explained by each principal component to aid in feature selection.

## Project Dependencies
To run the code successfully, the following Python libraries must be installed:
- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `scikit-learn` for model building and evaluation.
- `matplotlib` and `seaborn` for data visualization.
- `statsmodels` for calculating VIF.
- `scipy` for statistical functions.
- `xgboost` for advanced gradient boosting.

### Installation
Ensure all dependencies are installed by running the command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy xgboost
```

## How to Use This Code
1. **Data Preparation**: Ensure the dataset `openpowerlifting_data.csv` is available in the root directory of the project.
2. **Run the Code**:
   - Execute the Python script in an environment like Jupyter Notebook, Google Colab, or directly via a Python interpreter.
3. **Interpret Outputs**:
   - Review the correlation heatmap and VIF table for multicollinearity insights.
   - Evaluate regression model outputs for MAE, R² scores, and tolerance-based predictive accuracy.
   - Examine the scree plot and explained variance for PCA to understand feature importance.

## Code Walkthrough
### 1. **Data Loading and Preprocessing**
   - Reads the CSV dataset and selects relevant columns.
   - Encodes and filters the dataset for specific criteria (e.g., raw male lifters only).
   - Handles missing values and ensures appropriate data types for modeling.

### 2. **EDA (Exploratory Data Analysis)**
   - Generates a correlation heatmap to display relationships among predictors.
   - Calculates VIF values to check multicollinearity, crucial for reliable regression analysis.

### 3. **Model Training and Evaluation**
   - **Linear and Ridge Regression**:
     - Models train on predictors excluding the response variable, and results are evaluated for each lift.
   - **XGBoost and RandomForest**:
     - Advanced models train similarly, offering comparisons against linear models.
   - **Model Metrics**:
     - Displays MAE, R² scores, and tolerance accuracy for each model.

### 4. **PCA Analysis**
   - Conducts PCA on scaled predictor variables.
   - Displays explained variance ratios and cumulative variance for component selection.
   - Generates a scree plot for visual analysis of component importance.

## Results Interpretation
- **MAE and R²**:
  - Lower MAE indicates better predictive accuracy.
  - Higher R² shows the proportion of variance in the response variable explained by the model.
- **Tolerance Accuracy**:
  - Percentage of predictions within 10% of actual values, providing a real-world interpretation of model reliability.
- **PCA**:
  - Visualizes which components capture the most variance, aiding in feature reduction.

## Example Output
```text
Best3SquatKg Linear Regression - Mean Absolute Error: 15.23
Best3SquatKg Linear Regression - R² (R-squared): 0.78
Predictive accuracy of linear regression: 84.5%

Explained variance ratio: [0.45, 0.30, 0.15]
Cumulative explained variance: [0.45, 0.75, 0.90]
```

## Visualizations
- **Heatmap**: Shows correlations between age, body weight, and lift performances.
- **Scree Plot**: Illustrates the variance captured by each principal component, guiding feature selection.

## Limitations and Future Work
- The current model analysis is limited to raw male lifters; future iterations could extend to female and mixed categories.
- Hyperparameter tuning and cross-validation could enhance model performance.
- Further feature engineering could include additional metrics such as training history or competition level.

## Acknowledgments
- **OpenPowerlifting Project**: For providing the comprehensive dataset.


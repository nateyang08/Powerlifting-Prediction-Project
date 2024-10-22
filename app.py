import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as mp
import numpy as np
import seaborn as sb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import xgboost as xgb

# Load the dataset
data = pd.read_csv('openpowerlifting_data.csv')

# Select relevant columns and drop rows with missing values
filtered_data = data[['Sex', 'Equipment', 'Age', 'BodyweightKg', 
                  'Best3SquatKg', 'Best3BenchKg',
                  'Best3DeadliftKg', 'Place']].dropna()

# Numerical encoding of the categorical variables
filtered_data.loc[filtered_data.Sex == 'M', 'Sex'] = 1.0
filtered_data.loc[filtered_data.Sex == 'F', 'Sex'] = 2.0
filtered_data.loc[filtered_data.Sex == 'Mx', 'Sex'] = 3.0
filtered_data.loc[filtered_data.Equipment == 'Raw', 'Equipment'] = 1.0

# Get rid of disqualified lifters and guests
place_mask = (filtered_data['Place'].isin(['DQ', 'G', 'DD']))
filtered_data = filtered_data[~place_mask]

# Convert 'Place' to numeric, ignoring invalid parsing
filtered_data['Place'] = pd.to_numeric(filtered_data['Place'], errors='coerce')

# Further filter the dataframe for Raw male lifters
raw_male = filtered_data[(filtered_data['Sex'] == 1.0) & 
                       (filtered_data['Equipment'] == 1.0)].copy()

# Convert non-numeric columns to floats
raw_male['Sex'] = raw_male['Sex'].astype(float)
raw_male['Equipment'] = raw_male['Equipment'].astype(float)

# Correlation Matrix to check for multicollinearity
predictors = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']]
matrix = predictors.corr(numeric_only=True)
sb.heatmap(matrix, cmap='YlGnBu', annot=True)
mp.show()

# VIF Calculation to check multicollinearity
vif_data = pd.DataFrame()
vif_data['Features'] = predictors.columns
vif_data['VIF'] = [variance_inflation_factor(predictors.values, i)
                   for i in range(len(predictors.columns))]

# Response values for our different models 
response_values = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']

scaler = StandardScaler()

print('OpenPowerlifting Data Analysis: \n')

print('Linear Regression:')
# Linear Regression Loop
for response in response_values: 
    linear_X = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].drop(value, axis=1)
    linear_y = raw_male[response]

    X_train, X_test, y_train, y_test = train_test_split(linear_X, linear_y, test_size=0.2, random_state=42)

    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{response} Linear Regression - Mean Absolute Error: {mae}')
    print(f'{response} Linear Regression - R² (R-squared): {r2}')
    tolerance_range = 0.10 * y_test  # 10% tolerance range
    within_tolerance = np.abs(y_test - y_pred) <= tolerance_range
    tolerance_accuracy = np.mean(within_tolerance) * 100  
    print(f'Predictive accuracy of linear regression: {tolerance_accuracy} \n')

print('Ridge Regression:')
# Ridge Regression
for response in response_values: 
    ridge_X = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].drop(value, axis=1)
    ridge_y = raw_male[response]

    X_train, X_test, y_train, y_test = train_test_split(ridge_X, ridge_y, test_size=0.2, random_state=42)

    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Ridge model
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred = ridge_model.predict(X_test_scaled)

    # Evaluate Ridge model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{response} Ridge Regression - Mean Absolute Error: {mae}')
    print(f'{response} Ridge Regression - R² (R-squared): {r2}')
    tolerance_range = 0.10 * y_test  # 10% tolerance range
    within_tolerance = np.abs(y_test - y_pred) <= tolerance_range
    tolerance_accuracy = np.mean(within_tolerance) * 100  
    print(f'Predictive accuracy of ridge regression: {tolerance_accuracy}\n')

print('XG Boost:')
# XG Boost 
for response in response_values: 
    # Define the predictors and the target variable
    XG_X = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].drop(value, axis=1)
    XG_y = raw_male[response]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(XG_X, XG_y, test_size=0.2, random_state=42)

    # Standardize the features (optional for tree-based models, but keeping consistency)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the XGBoost Regressor
    xgboost_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

    # Train the model
    xgboost_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = xgboost_model.predict(X_test_scaled)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate tolerance accuracy
    tolerance_range = 0.10 * y_test  # 5% tolerance range
    within_tolerance = np.abs(y_test - y_pred) <= tolerance_range
    tolerance_accuracy = np.mean(within_tolerance) * 100  # Convert to percentage

    # Print the evaluation metrics
    print(f'{response} XGBoost prediction - Mean Absolute Error: {mae}')
    print(f'{response} XGBoost prediction - R² (R-squared): {r2}')
    print(f'{response} prediction - {0.10 * 100}% Tolerance Accuracy: {tolerance_accuracy:.2f}% \n')

# Principal Component Analysis (PCA)
features = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']]
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(features_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

print(f"Explained variance ratio: {explained_variance}")
print(f"Cumulative explained variance: {cumulative_variance}\n")

# Scree Plot 
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

print('RandomForest:')
# RandomForest
for value in response: 
    forest_X_male = raw_male[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].drop(value, axis=1)
    y_male = raw_male[value]

    X_train, X_test, y_train, y_test = train_test_split(forest_X_male, y_male, test_size=0.2, random_state=42)

    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    randomforest = RandomForestRegressor(n_estimators=100, random_state=42)
    randomforest.fit(X_train_scaled, y_train)
    y_pred = randomforest.predict(X_test_scaled)

    # Evaluate RandomForest model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{value} RandomForest prediction - Mean Absolute Error: {mae}')
    print(f'{value} RandomForest prediction - R² (R-squared): {r2} \n')





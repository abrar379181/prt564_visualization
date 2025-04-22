# **Linear Regression & Logistic Regression Model Evaluation**"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Prepare the data for modeling
# For Linear Regression, we will use 'Outbreak Logistics Regression' as the target variable
X = data[['Temparature', 'Poultry Density']]  # Features
y_linear = data['Outbreak Logistics Regression']  # Target variable for linear regression

# For Logistic Regression, we will create a binary target for 'Cases' (1 if there is an outbreak, 0 if not)
y_logistic = (data['Cases'] > 1).astype(int)  # 1 if Cases > 1, otherwise 0

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train_linear, y_test_linear = train_test_split(X, y_linear, test_size=0.2, random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_logistic, test_size=0.2, random_state=42)

# Scale the features for the regression models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, y_train_linear)
y_pred_linear = linear_reg_model.predict(X_test_scaled)

# Compute Mean Squared Error for the Linear Regression model
mse = mean_squared_error(y_test_linear, y_pred_linear)

# Logistic Regression Model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train_log)
y_pred_log = log_reg_model.predict(X_test_scaled)

# Compute Accuracy for the Logistic Regression model
log_acc = accuracy_score(y_test_log, y_pred_log)

mse, log_acc




# **Linear Regression Model Implementation**

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare the dataset
X = data[['Temparature', 'Poultry Density']]  # Features
y_linear = data['Outbreak Logistics Regression']  # Target variable for linear regression

# Split the dataset into training and testing sets
X_train, X_test, y_train_linear, y_test_linear = train_test_split(X, y_linear, test_size=0.2, random_state=42)

# Scale the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, y_train_linear)

# Generate predictions and compute Mean Squared Error (MSE)
y_pred_linear = linear_reg_model.predict(X_test_scaled)
mse = mean_squared_error(y_test_linear, y_pred_linear)

print(f'Mean Squared Error (Linear Regression): {mse}')




# **Logistic Regression Model Implementation**

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare the target variable
y_logistic = (data['Cases'] > 1).astype(int)  # Binary target variable (1 for Outbreak, 0 for No Outbreak)

# Split the dataset into training and testing subsets
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_logistic, test_size=0.2, random_state=42)

# Scale the feature data
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

# Fit the Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train_log)

# Generate predictions and compute the accuracy
y_pred_log = log_reg_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test_log, y_pred_log)

print(f'Accuracy of Logistic Regression: {log_acc}')




# **Statistical Test Results of t-statistic & p-value**

# Import the required libraries and reload the dataset
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Reload the dataset
data = pd.read_csv('/content/sample_data/avian_data.csv')

# Prepare the features and target variables for modeling
features = data[['Temparature', 'Poultry Density']]  # Features

# Define the target variable for Linear Regression
target_linear = data['Outbreak Logistics Regression']  # Target for linear regression

# Create a binary target variable for Logistic Regression based on 'Cases'
target_logistic = (data['Cases'] > 1).astype(int)  # 1 if Cases > 1, otherwise 0

# Split the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train_linear, y_test_linear = train_test_split(features, target_linear, test_size=0.2, random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(features, target_logistic, test_size=0.2, random_state=42)

# Standardize the feature data for the regression models
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Fit the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_standardized, y_train_linear)
predictions_linear = linear_model.predict(X_test_standardized)

# Fit the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_standardized, y_train_log)
predictions_logistic = logistic_model.predict(X_test_standardized)

# Calculate the residuals for both models
residuals_linear = y_test_linear - predictions_linear
residuals_logistic = y_test_log - predictions_logistic

# Conduct a paired t-test on the residuals
t_statistic, p_value = ttest_rel(residuals_linear, residuals_logistic)

t_statistic, p_value
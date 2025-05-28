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
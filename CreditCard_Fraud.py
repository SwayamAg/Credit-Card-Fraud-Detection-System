# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset (ensure you have the dataset available)
data = pd.read_csv("creditcard.csv")

# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop(['Class'], axis=1)  # Features (all columns except 'Class')
y = data['Class']  # Target (Fraud: 1, Non-fraud: 0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Count the number of fraud vs non-fraud transactions
sns.countplot(x='Class', data=data)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm_r', annot_kws={'size': 20})
plt.title('Correlation Heatmap')
plt.show()

# Import model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Logistic Regression:")
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree:")
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# Accuracy Comparison
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred)}")

# TOOK AROUND 4'18''
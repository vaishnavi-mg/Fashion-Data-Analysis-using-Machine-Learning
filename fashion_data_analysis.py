import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the Dataset
file_path = '/content/myntra_products_catalog.csv'
df = pd.read_csv(file_path)

# Verify the column names
print(df.columns)

# Step 2: Data Visualization
# Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.show()

# Product Brand Distribution
plt.figure(figsize=(10, 8))
sns.countplot(data=df, y='ProductBrand', order=df['ProductBrand'].value_counts().index)
plt.title('Top Brands Distribution')
plt.show()

# Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Price (INR)'], kde=True)
plt.title('Price Distribution')
plt.show()

# Step 3: Feature Selection
# Convert categorical data to numerical
label_encoder = LabelEncoder()
df['ProductName'] = label_encoder.fit_transform(df['ProductName'])
df['ProductBrand'] = label_encoder.fit_transform(df['ProductBrand'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Drop the 'Description' column and apply OneHotEncoder to 'PrimaryColor'
df = df.drop(columns=['Description'])
df = pd.get_dummies(df, columns=['PrimaryColor'], drop_first=True)

# Split features and target
X = df.drop(columns=['Price (INR)'])
y = df['Price (INR)']

# Feature Selection using SelectKBest
select_k_best = SelectKBest(score_func=chi2, k=3)
X_new_kbest = select_k_best.fit_transform(X, y)
print(f"Selected features using SelectKBest: {X.columns[select_k_best.get_support()]}")

# Feature Selection using RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
X_new_rfe = rfe.fit_transform(X, y)
print(f"Selected features using RFE: {X.columns[rfe.get_support()]}")

# Step 4: Classification Algorithms
# Without Feature Selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, y_pred_log_reg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy without Feature Selection: {log_reg_acc}")
print(f"Random Forest Accuracy without Feature Selection: {rf_acc}")

# With Feature Selection (Using SelectKBest)
X_train_kbest, X_test_kbest, y_train_kbest, y_test_kbest = train_test_split(X_new_kbest, y, test_size=0.2, random_state=42)

# Logistic Regression with SelectKBest
log_reg.fit(X_train_kbest, y_train_kbest)
y_pred_log_reg_kbest = log_reg.predict(X_test_kbest)
log_reg_acc_kbest = accuracy_score(y_test_kbest, y_pred_log_reg_kbest)

# Random Forest with SelectKBest
rf.fit(X_train_kbest, y_train_kbest)
y_pred_rf_kbest = rf.predict(X_test_kbest)
rf_acc_kbest = accuracy_score(y_test_kbest, y_pred_rf_kbest)

print(f"Logistic Regression Accuracy with SelectKBest: {log_reg_acc_kbest}")
print(f"Random Forest Accuracy with SelectKBest: {rf_acc_kbest}")

# Step 5: Compare the Results
print("Performance Comparison:")
print(f"Logistic Regression: Before Feature Selection: {log_reg_acc}, After Feature Selection: {log_reg_acc_kbest}")
print(f"Random Forest: Before Feature Selection: {rf_acc}, After Feature Selection: {rf_acc_kbest}")

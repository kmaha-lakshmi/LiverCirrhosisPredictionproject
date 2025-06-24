import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the Excel file
df = pd.read_excel('indian_liver_patient.csv.xlsx')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print("Columns in dataset:", df.columns.tolist())
# Features and target
X = df.drop('Selector', axis=1)  # 'Selector' is the target column
y = df['Selector']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open('rf_acc_68.pkl', 'wb'))
pickle.dump(scaler, open('normalizer.pkl', 'wb'))

print("✅ Model and scaler saved successfully.")

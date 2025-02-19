import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

# Load the dataset
file_path = "/mnt/data/pml-training (1).csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
irrelevant_columns = [
    "Unnamed: 0", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window"
]
df_cleaned = df.drop(columns=irrelevant_columns)

# Convert all features to numeric, coercing errors to NaN
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')

# Reintroduce target variable
df_cleaned["classe"] = df["classe"]

# Drop columns with excessive missing values (threshold: 75% non-null values required)
df_cleaned = df_cleaned.dropna(axis=1, thresh=0.75 * len(df_cleaned))

# Split dataset into features and target variable
X = df_cleaned.drop(columns=["classe"])
y = df_cleaned["classe"]

# Split into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the Random Forest model with parallel processing
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")

# Train the model on the full training set
rf_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Select 20 random test cases for prediction
random_samples = X_test.sample(n=20, random_state=42)
predictions = rf_model.predict(random_samples)

# Print results
print("Model Performance:")
print(f" - Cross-validation Accuracy: {np.mean(cv_scores):.4f}")
print(f" - Test Set Accuracy: {test_accuracy:.4f}")
print("\nPredictions for 20 test cases:")
print(predictions)

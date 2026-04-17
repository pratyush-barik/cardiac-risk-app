import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("heart disease dataset.csv")

# Features & target
X = df.drop("Heart_Risk", axis=1)
y = df["Heart_Risk"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=5)

# Train models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Save models
joblib.dump(lr, "model/lr.pkl")
joblib.dump(dt, "model/dt.pkl")
joblib.dump(rf, "model/rf.pkl")
joblib.dump(knn, "model/knn.pkl")

print("Models trained and saved successfully ")

from sklearn.metrics import accuracy_score, classification_report

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
knn_pred = knn.predict(X_test)

# Accuracy
print("\n=== MODEL ACCURACY ===")
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("KNN:", accuracy_score(y_test, knn_pred))
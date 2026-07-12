import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("gesture_data.csv", header=None)

# First column = labels
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")
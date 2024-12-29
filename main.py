import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Preprocessing: Splitting and Scaling
X = data.iloc[:, :-1]
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualization: Confusion Matrix
plot_confusion_matrix(model, X_test, y_test, display_labels=iris.target_names, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Prediction Example
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
predicted_class = iris.target_names[prediction[0]]
print(f"Predicted Class: {predicted_class}")

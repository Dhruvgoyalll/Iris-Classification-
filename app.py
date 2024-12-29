import streamlit as st
import pandas as pd
from scikit_learn.datasets import load_iris
from scikit_learn.tree import DecisionTreeClassifier
from scikit_learn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Train model
X = data.iloc[:, :-1]
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_scaled, y)

# Streamlit app
st.title("Iris Flower Prediction")

# Input fields for user
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    st.write(f"Predicted Class: {iris.target_names[prediction[0]]}")

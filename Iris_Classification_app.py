import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
import os


# Load model using pickle
@st.cache_resource
def load_model():
    """Load the trained model."""
    with open('iris_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(f"Model loaded successfully: {type(model)}")

# Load the dataset from a local file
@st.cache_data
def load_dataset():
    """Load the Iris dataset from a local CSV file."""
    iris_df = pd.read_csv("Iris.csv")
    # Ensure correct column naming
    if 'Species' not in iris_df.columns:
        iris_df.rename(columns={'species': 'Species'}, inplace=True)
    return iris_df

# Load the model and dataset
model = load_model()
iris_df = load_dataset()

# Streamlit UI components
st.title("Iris Flower Classification")
st.write("This app predicts the species of an Iris flower based on its measurements using a pre-trained model.")

# Sidebar for user input
st.sidebar.header("Input Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(iris_df['SepalLengthCm'].min()), float(iris_df['SepalLengthCm'].max()), 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()), 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()), 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()), 1.3)

#User input as a DataFrame with placeholder Id
user_input = pd.DataFrame({
    'Id': [1],  # Placeholder, ensure it's consistent with training data
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})


st.subheader("User Input Measurements")
st.write(user_input)

# Predict the species
if st.button("Predict Species"):
    prediction = model.predict(user_input)
    predicted_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}[prediction[0]]
    st.success(f"The predicted species is **{predicted_species}**.")

# Visualize dataset
st.subheader("Explore the Iris Dataset")
st.write("Here's a preview of the dataset used for training the model:")
st.dataframe(iris_df)

# Footer
st.write("---")
st.markdown("<p style='text-align: center;'>&copy; 2025 @suraj_nate All rights reserved.</p>", unsafe_allow_html=True)

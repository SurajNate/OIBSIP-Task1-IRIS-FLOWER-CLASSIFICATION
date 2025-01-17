import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the model
@st.cache_resource
def load_model():
    """Load the trained model."""
    model = joblib.load("Jupyter Files/iris_classifier_model.pkl")
    st.write(type(model))  # Check the model's type
    return model

# Load the dataset from a local file
@st.cache_data
def load_dataset():
    """Load the Iris dataset from a local CSV file."""
    iris_df = pd.read_csv("Jupyter Files/Iris.csv")
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

# User input as a DataFrame with placeholder Id
user_input = pd.DataFrame({
    'Id': [1],
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})

st.subheader("User Input Measurements")
st.write(user_input)

# Remove the 'Id' column from the input data before prediction
user_input = user_input.drop('Id', axis=1)

# Predict the species using the model's predict() method
if st.button("Predict Species"):
    prediction = model.predict(user_input)  # Using the 'predict' method of the model
    st.success(f"The predicted species is : **{prediction[0]}**.")

# Visualize dataset
st.subheader("Explore the Iris Dataset")
st.write("Here's a preview of the dataset used for training the model:")
st.dataframe(iris_df)

# Footer
st.write("---")
st.markdown("<p style='text-align: center;'>&copy; 2025 @suraj_nate All rights reserved.</p>", unsafe_allow_html=True)

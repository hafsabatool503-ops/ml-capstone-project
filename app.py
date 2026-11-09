import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model/model.pkl', 'rb'))

st.title("🚀 End-to-End Machine Learning Capstone App")
st.write("Enter the feature values below to get predictions.")

# Input fields
sepal_length = st.number_input('Sepal Length', value=5.1)
sepal_width = st.number_input('Sepal Width', value=3.5)
petal_length = st.number_input('Petal Length', value=1.4)
petal_width = st.number_input('Petal Width', value=0.2)

if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f'✨ Predicted Class / Output: {prediction[0]}')

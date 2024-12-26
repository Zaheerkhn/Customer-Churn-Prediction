import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import joblib

# Load the trained model without compiling
try:
    model = tf.keras.models.load_model('Artifacts/customer_churn_model.h5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the encoders and scalers
try:
    scaler = joblib.load('Artifacts/scaler.pkl')
    label_encoder = joblib.load('Artifacts/label_encoder.pkl')
    one_hot_encoder = joblib.load('Artifacts/one_hot_encoder.pkl')
except Exception as e:
    st.error(f"Error loading encoders or scalers: {e}")
    st.stop()

# Title
st.title('Customer Churn Prediction')

# User input
# Create columns
col1, col2, col3, col4 = st.columns(4)

# Row 1
with col1:
    geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
with col2:
    gender = st.selectbox('Gender', label_encoder.classes_)
with col3:
    age = st.text_input('Age', value="18")
with col4:
    tenure = st.slider('Tenure', 0, 10)

# Row 2
with col1:
    balance = st.text_input('Balance', value="0.0")
with col2:
    products = st.slider('Number of products', 1, 4)
with col3:
    credit_card = st.selectbox('Has credit card', ['yes', 'no'])
    credit_card_mapping = {'yes': 1, 'no': 0}
    credit_card_value = credit_card_mapping[credit_card]
with col4:
    active_member = st.selectbox('Is active member', ['yes', 'no'])
    active_member_mapping = {'yes': 1, 'no': 0}
    active_member_value = active_member_mapping[active_member]

# Row 3
with col1:
    salary = st.text_input('Estimated Salary', value="0.0")
with col2:
    creditscore = st.text_input('Credit Score', value="300")

# Prepare the input data
def preprocess_and_predict():
    try:
        input_data = pd.DataFrame({
            'creditscore': [float(creditscore)],
            'gender': [label_encoder.transform([gender])[0]],
            'age': [int(age)],
            'tenure': [tenure],
            'balance': [float(balance)],
            'numofproducts': [products],
            'hascrcard': [credit_card_value],
            'isactivemember': [active_member_value],
            'estimatedsalary': [float(salary)]
        })
    except ValueError:
        st.error("Please enter valid numeric values for Age, Balance, Credit Score, and Estimated Salary.")
        return

    # One hot encoding geography
    geo_encoder = one_hot_encoder.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder.get_feature_names_out(['geography']))

    # Combining data
    input_data = pd.concat([input_data, geo_df], axis=1)

    # Scaling the data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn')
    else:
        st.write('The customer is not likely to churn')
    st.write('Probability of churning: ', prediction_proba)

# Button to perform prediction
if st.button('Predict'):
    preprocess_and_predict()

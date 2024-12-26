import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('Artifacts/model.h5')

# Load the encoders and scalers
with open('Artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('Artifacts/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('Artifacts/one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

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
    age = st.number_input('Age', min_value=18, max_value=100)
with col4:
    tenure = st.slider('Tenure', 0, 10)
# Row 2
with col1:
    balance = st.number_input('Balance', min_value=0.0)
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
    salary = st.number_input('Estimated Salary', min_value=0.0)
with col2:
    creditscore = st.number_input('Credit Score', min_value=300, max_value=850)

# Prepare the input data
def preprocess_and_predict():
    input_data = pd.DataFrame({
        'creditscore': [creditscore],
        'gender': [label_encoder.transform([gender])[0]],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'numofproducts': [products],
        'hascrcard': [credit_card_value],
        'isactivemember': [active_member_value],
        'estimatedsalary': [salary]
    })

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

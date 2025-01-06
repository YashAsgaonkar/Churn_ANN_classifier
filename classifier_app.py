import tensorflow as tf
import numpy as np
import pandas as pd
import pickle as pkl
from tensorflow.keras.models import load_model
import streamlit as st
# Load the model
model = load_model('model.h5')

# Load the scalers and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    gen_encoder = pkl.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    geo_encoder = pkl.load(f)

## streamlit app
st.title('Customer Churn Prediction')
st.write('This is a simple app to predict customer churn using a neural network model.')

# Input form
st.sidebar.header('User Input Parameters')

credit_score = st.sidebar.slider('Credit Score', 350, 850, 650)
geography = st.sidebar.selectbox('Geography', geo_encoder.categories_[0])
estimated_salary = st.sidebar.number_input('Estimated Salary')
age = st.sidebar.slider('Age', 18, 92, 45)
balance = st.sidebar.number_input('Balance')
products = st.sidebar.slider('Number of Products', 1, 4, 2)
has_card = st.sidebar.selectbox('Has Credit Card', [0,1])
is_active = st.sidebar.selectbox('Is Active Member',[0,1])
tenure = st.sidebar.slider('Tenure', 0, 10, 5)
gender = st.sidebar.selectbox('Gender',gen_encoder.classes_)

if st.sidebar.button('Submit'):
    input_data = {
        'CreditScore': credit_score,
        'Gender': gen_encoder.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Geography': geography
    }

    # do ohc on the geography
    geo_encoded = geo_encoder.transform([[input_data['Geography']]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.categories_[0])

    ##Combine the input data with the one hot encoded geography
    input_df = pd.DataFrame([input_data])
    input_df.drop('Geography', axis=1, inplace=True)
    input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

    ## Error observed due to my inconsistency in the column names (used .categories_[0] instead)
    input_df.rename(columns={'France': 'Geography_France', 'Germany': 'Geography_Germany', 'Spain': 'Geography_Spain'}, inplace=True)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    if prediction:
        st.subheader('Prediction')
        if prediction > 0.5:
            st.write('The customer is likely to churn')
        else:
            st.write('The customer is not likely to churn')

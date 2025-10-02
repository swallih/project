import streamlit as st
import numpy as np

import joblib
model=joblib.load("ml_project.pkl")
encoder1=joblib.load("le.pkl")
scaler=joblib.load("scaler.pkl")
st.title("Rainfall Prediction Model")
st.write("Forecasting rain based on weather conditions and environmental data")

Location = st.selectbox("Select Location",['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix','Philadelphia'])
Temperature=st.number_input("enter temperature")
Humidity=st.number_input("enter humidity")
Wind_Speed=st.number_input("enter wind_speed")
Precipitation=st.number_input("enter precipitation")
Cloud_Cover	=st.number_input("enter cloud cover")
Pressure=st.number_input("enter pressure")

location=encoder1.fit_transform([Location])[0]

if st.button("predict"):
    scaled_features = scaler.transform([[Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Pressure]])
    input_features = np.concatenate([[location], scaled_features[0]])  # Combine label-encoded location with scaled values
    result = model.predict([input_features])[0]
    st.success("the output is {}".format(result))
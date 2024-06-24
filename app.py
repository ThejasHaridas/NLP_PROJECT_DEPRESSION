import streamlit as st
import joblib

model = joblib.load("depression-classification.pkl")

Labels = {'moderate':'Moderately Depressed', 'not depression':'Not Depressed','severe':'Severely Depressed'}

st.title("Depression Analysis from Social Media Text")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    predicted_sentiment = model.predict([user_input])[0]
    predicted_sentiment_label = Labels[str(predicted_sentiment)]
    st.info(f"Predicted that the person is: {predicted_sentiment_label}")
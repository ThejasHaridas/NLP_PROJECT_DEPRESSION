import streamlit as st
import joblib

model=joblib.load("classification_depression.pkl")

labels = {'moderate': 'moderately', 'not depression': 'not', 'severe': 'severely'}

st.title("Depression Analysis From Social Media Text")

user_input=st.text_area("enter your text here")

if st.button("predict"):
    predicted_label = model.predict([user_input])[0]
    predicted_depression_label = labels[str(predicted_label)]
    st.info(f"predicted that the person is {predicted_depression_label} depressed")

import streamlit as st
import pickle
model = pickle.load(open("model.pkl", "rb"))
st.title("Text-Analytics CA-2")
review = st.text_area("Enter Movie Review")
if st.button("Predict"):
    prediction = model.predict([review])
    st.success(f"Sentiment : {prediction[0]}")

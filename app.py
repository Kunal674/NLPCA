import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Text-Analytics CA2",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 IMDB Sentiment Analysis")
st.write("Predict whether a movie review is Positive or Negative.")

# Input box
review = st.text_area(
    "Enter Movie Review",
    placeholder="Example: This movie was absolutely amazing with great acting and storyline!"
)

# Prediction button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    
    else:
        prediction = model.predict([review])

        if prediction[0] == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

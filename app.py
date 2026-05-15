import streamlit as st
import pickle

# Load trained model & vectorizer
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineScope AI",
    page_icon="🎬",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

    .main {
        background: linear-gradient(to bottom right, #0f172a, #111827);
        color: white;
    }

    .title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 35px;
    }

    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border: 2px solid #ff4b4b;
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
    }

    .stButton button {
        width: 100%;
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton button:hover {
        transform: scale(1.02);
        opacity: 0.95;
    }

    .positive-box {
        background-color: #14532d;
        color: #bbf7d0;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 25px;
    }

    .negative-box {
        background-color: #7f1d1d;
        color: #fecaca;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 25px;
    }

    .footer {
        text-align: center;
        color: gray;
        margin-top: 50px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🎬 CineScope AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------
review = st.text_area(
    "Write your movie review below:",
    height=220,
    placeholder="Example: The movie was visually stunning and emotionally powerful..."
)

# ---------------- BUTTON ----------------
if st.button("✨ Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        review_vector = tfidf.transform([review])

        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.markdown(
                '<div class="positive-box">😊 Positive Sentiment</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="negative-box">😔 Negative Sentiment</div>',
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Built using NLP • TF-IDF • Logistic Regression</div>',
    unsafe_allow_html=True
)

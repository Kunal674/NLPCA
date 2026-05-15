import re
import pickle
import pandas as pd
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CinePulse AI",
    page_icon="🎥",
    layout="centered"
)

# -----------------------------
# CUSTOM UI
# -----------------------------
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
        }

        .main {
            background-color: #0E1117;
            color: white;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #FF4B4B;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #BBBBBB;
            margin-bottom: 30px;
        }

        .stTextArea textarea {
            background-color: #1E1E1E;
            color: white;
            border-radius: 12px;
            border: 1px solid #FF4B4B;
        }

        .stButton button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            border: none;
            padding: 10px;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }

        .positive {
            background-color: #163d1d;
            color: #7CFC98;
        }

        .negative {
            background-color: #3d1616;
            color: #ff7b7b;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">🎥 CinePulse AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Smart Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

# -----------------------------
# NLP SETUP
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")

# -----------------------------
# TEXT CLEANING
# -----------------------------
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', ' ', text)

    text = re.sub(r'http\S+|www\S+', ' ', text)

    text = re.sub(r'\d+', ' ', text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_words)

df['processed_review'] = df['review'].apply(clean_text)

# -----------------------------
# LABEL ENCODING
# -----------------------------
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

X = df['processed_review']
y = df['sentiment']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000
)

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
sentiment_model = LogisticRegression(max_iter=1000)

sentiment_model.fit(X_train_vectorized, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = sentiment_model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(sentiment_model, open("cinepulse_model.pkl", "wb"))

pickle.dump(vectorizer, open("cinepulse_vectorizer.pkl", "wb"))

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.write("### ✍️ Enter a Movie Review")

user_review = st.text_area(
    "",
    height=180,
    placeholder="Type your movie review here..."
)

if st.button("Analyze Sentiment"):

    if user_review.strip() == "":
        st.warning("Please enter a review first!")

    else:
        cleaned_review = clean_text(user_review)

        review_vector = vectorizer.transform([cleaned_review])

        prediction = sentiment_model.predict(review_vector)[0]

        if prediction == 1:
            st.markdown(
                '<div class="result-box positive">😊 Positive Review</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box negative">😔 Negative Review</div>',
                unsafe_allow_html=True
            )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <br>
    <center>
        <p style='color:gray;'>
            Built with ❤️ using NLP, TF-IDF & Logistic Regression
        </p>
    </center>
    """,
    unsafe_allow_html=True
)

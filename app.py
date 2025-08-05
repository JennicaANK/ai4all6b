import streamlit as st
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

@st.cache_resource
def load_model():
    return joblib.load("xgb_price_model.pkl")

@st.cache_data
def load_rolling():
    return np.load("mean_rolling_features.npy")

@st.cache_resource
def load_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    return tokenizer, model

model = load_model()
rolling_avg = load_rolling().reshape(1, -1)
tokenizer, finbert_model = load_tokenizer_model()

keywords = ['earnings', 'tech', 'best', 'netflix', 'stocks', 'buy', 'apple', 'stock', 'growth', 'market']

def predict_price_direction(headline):
    headline_lower = headline.lower()
    keyword_vector = np.array([headline_lower.count(word) for word in keywords]).reshape(1, -1)

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        sentiment_score = (probs[2] * 1 + probs[1] * 0 + probs[0] * -1).item()

    sentiment_score_array = np.array([[sentiment_score]])
    final_input = np.hstack((keyword_vector, sentiment_score_array, rolling_avg))
    prediction = model.predict(final_input)

    return "📈 Price Up" if prediction[0] == 1 else "📉 Price Down"

st.set_page_config(page_title="FAANG Stock Predictor", page_icon="📊")
st.title("📊 Stock Price Direction Predictor")
st.markdown("Enter a financial news headline to see if the stock is expected to go **up** or **down**.")

headline = st.text_input("📰 Enter a news headline")
if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        prediction = predict_price_direction(headline)
        st.success(f"Prediction: **{prediction}**")

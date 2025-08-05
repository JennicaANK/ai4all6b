# Predicting Stock Market Prices Using Newspaper Headlines  
**AI4ALL Ignite Project | Group 6B**  
**Contributors:** Griheet Tandra, Nargiz Akhmetova, Ikhlas Palakkattu, Mills Yamoah, Aye Nyein Kyaw  

---

##  Project Summary  
Predicted short-term stock price movements (up/down trends) using daily financial news headlines by combining natural language processing and machine learning techniques.  
Applied FinBERT and XGBoost to model sentiment-driven stock fluctuations across FAANG stocks—leveraging real-world financial datasets during the AI4ALL Ignite accelerator.

---

##  Problem Statement  
As real-time financial headlines increasingly drive market sentiment, investors—especially retail traders—lack tools to systematically interpret this impact.  
This project investigates whether news sentiment and keyword patterns can accurately predict short-term stock price directions for major tech companies (FAANG), enhancing data-driven decision-making.

---

##  Key Results  
-  Processed 10,000+ headlines and aligned them with FAANG stock movements  
-  Identified key biases in financial news (sentiment ambiguity, lag effects, non-financial influences)  
-  Achieved over **70% test accuracy** using XGBoost for binary classification (PriceUp = 1/-1)  
-  Created real-time, user-friendly prediction interface with Streamlit deployment

---

## ⚙ Methodologies  
- **FinBERT Sentiment Extraction**: Used a pre-trained financial transformer model to assign sentiment scores to each headline (positive, neutral, negative).  
- **XGBoost Classification**: Engineered features like keyword frequency and rolling stock returns to classify next-day price direction.  
- **Streamlit App**: Built an interactive web UI for headline input and live prediction using the trained XGBoost model.

---

##  Data Sources  
-  [Kaggle Dataset by Kaushik Suresh](https://www.kaggle.com/datasets/kaushiksuresh147/faang-stock-data) – Historical FAANG stock prices  
-  [Kaggle Dataset by Skywalker290](https://www.kaggle.com/datasets/saife245/news-headlines-dataset-for-stock-sentiment-analysis) – Financial news articles  
-  [FNSPID Financial News Dataset (GitHub)](https://github.com/gdmarmerola/FNSPID) – Supplementary news dataset with target signals

---

##  Technologies Used  
- Python  
- pandas, NumPy  
- FinBERT (Transformers via HuggingFace)  
- XGBoost  
- Streamlit  
- scikit-learn  

---

##  Success Criteria  
1. **Predictive Accuracy**  
   - ≥ 70% accuracy in predicting FAANG stock up/down direction  
   - Metrics: Accuracy, Recall, F1 Score  

2. **Consistency Over Time**  
   - Stable performance across earnings and non-earnings periods (via time-based CV)  

4. **Real-World Utility**  
   - Built a deployable Streamlit interface suitable for fintech analysts and retail investors  

---

##  Sources & Inspiration  
- [How the News Affects Stock Prices](https://www.investopedia.com/ask/answers/155.asp)  
- [FinBERT: Financial Sentiment Analysis](https://arxiv.org/abs/1908.10063)  
- [Level Fields AI: News Impact on Stocks](https://www.levelfields.ai/news/impact-of-news-on-stock-prices)  
- [Stock Price Prediction Using Transformers](https://medium.com/@Matthew_Frank/stock-price-prediction-using-transformers-2d84341ff213)  
- [NLP in Finance (Medium)](https://leomercanti.medium.com/natural-language-processing-nlp-in-finance-how-ai-is-transforming-market-analysis-9b7c4d2c5c61)  
- [Lazy Programmer: Predict Stocks with News](https://lazyprogrammer.me/free-exercise-predict-stocks-with-news-other-ml-news/)

---


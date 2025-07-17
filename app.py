import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="🕵 Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news article below to find out if it's REAL or FAKE.")

# Input section
title = st.text_input("📌 News Title", placeholder="e.g. Scientists discover water on Mars")
text = st.text_area("✍ News Content", height=200, placeholder="Enter the full article content here...")
content = title + " " + text

# Predict button
if st.button("🔍 Predict"):
    if not content.strip():
        st.warning("Please enter both title and content.")
    else:
        # Vectorize and predict
        transformed = vectorizer.transform([content])
        prediction = model.predict(transformed)[0]
        score = model.decision_function(transformed)[0]
        confidence = round(min(abs(score) / 5 * 100, 100), 2)  # scale confidence (not probability)

        # Styled output
        if prediction == "FAKE":
            st.error(f"🔴 This news is likely FAKE")
        else:
            st.success(f"🟢 This news is likely REAL")

        st.info(f"📊 Confidence Score: *{confidence}%*")

        # WordCloud
        st.subheader("🧾 Word Cloud of Submitted News")
        wc = WordCloud(width=800, height=400, background_color='white').generate(content)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Export result
        if st.button("📁 Export this prediction"):
            df = pd.DataFrame([[title, text, prediction]], columns=["Title", "Content", "Prediction"])
            df.to_csv("prediction_output.csv", index=False)
            st.success("✅ Saved as prediction_output.csv")

# Footer
st.markdown("---")
st.caption("Built with ❤ using Streamlit and scikit-learn.")
# 📰 Fake News Detector 🕵️‍♂️

This is a simple web app built with **Streamlit** that detects whether a news article is **Real** or **Fake** using a trained machine learning model. Just enter the news title and content, and the app will predict the authenticity of the article along with a confidence score and a word cloud of the input.

---

## 🚀 Features

- ✅ Detect if news is **FAKE** or **REAL**
- 📊 Show a **confidence score** for the prediction
- ☁️ Generate a **word cloud** of your news content
- 💾 Option to **export** the prediction as a CSV file
- 🎨 Clean and intuitive Streamlit interface

---

## 🧠 How It Works

1. The news **title** and **content** are combined and vectorized using a TF-IDF vectorizer.
2. A trained machine learning model classifies the text as `FAKE` or `REAL`.
3. The result is shown along with a confidence score and a word cloud visualization.
4. Users can optionally save the prediction result as a CSV file.

---

## 📂 Project Structure


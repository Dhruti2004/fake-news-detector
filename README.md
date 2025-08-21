# 📰 Fake News Detector 🕵️‍♂️

This is a simple web app built with **Streamlit** that detects whether a news article is **Real** or **Fake** using a trained machine learning model. Just enter the news title and content, and the app will predict the authenticity of the article along with a confidence score and a word cloud of the input.

---

## 🔎 Overview

- Fake news spreads quickly online and can cause real-world consequences.  
- This project uses **TF-IDF vectorization** and a **PassiveAggressiveClassifier** to classify news headlines.  
- The app provides an **interactive interface** where users can test single headlines or evaluate a batch of headlines via CSV.

---

## 🚀 Features

✔️ Predict if a single news headline is **REAL** or **FAKE**  
✔️ Show confidence score for predictions  
✔️ Generate a **Word Cloud** of the entered headline  
✔️ Upload a CSV file (`Title`, `Label`) for bulk evaluation  
✔️ Display **Accuracy, Precision, Recall, and F1-Score**  
✔️ Export predictions into a CSV file  
✔️ Generate and export a **detailed evaluation report** (metrics table)  

---

## 🧠 How It Works

1. The news **title** and **content** are combined and vectorized using a **TF-IDF vectorizer**.  
2. A trained **machine learning model** classifies the text as `FAKE` or `REAL`.  
3. The result is shown along with a **confidence score** and a **word cloud visualization**.  
4. Users can optionally **save the prediction result** as a CSV file.  
5. For **bulk evaluation**, users can upload a CSV file (`Title`, `Label`) → the app calculates **Accuracy, Precision, Recall, and F1-Score**, and allows exporting results.  
6. A **classification report** (detailed metrics table) can also be exported for documentation.  

---

## 📷 Screenshots

### 🏠 Home Page
<img width="1190" height="350" alt="Home" src="https://github.com/user-attachments/assets/c65bbd17-9541-4ddf-82a7-2d9acacef425" />

### 🔍 Single Prediction
<img width="1190" height="350" alt="Single_prediction" src="https://github.com/user-attachments/assets/4619e278-e395-4ccf-8d70-8c87ef13ff19" />

### ☁ Word Cloud
<img width="1190" height="350" alt="wordcloud" src="https://github.com/user-attachments/assets/31da86d6-b5ba-4078-bc63-09a319d0102c" />

### 📂 CSV Evaluation
<img width="1190" height="370" alt="evaluation" src="https://github.com/user-attachments/assets/9d1488ee-6c07-4182-89e3-8808b356fb7c" />

---

✨ Built with ❤ using **Streamlit** and **Scikit-learn**

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="🕵 Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news headline below to find out if it's REAL or FAKE.")

# =========================
# 🔹 Single Title Prediction
# =========================
title = st.text_input("📌 News Title", placeholder="e.g. Scientists discover water on Mars")

if st.button("🔍 Predict"):
    if not title.strip():
        st.warning("Please enter a news title.")
    else:
        # Vectorize and predict
        transformed = vectorizer.transform([title])
        prediction = model.predict(transformed)[0]
        score = model.decision_function(transformed)[0]
        confidence = round(min(abs(score) / 5 * 100, 100), 2)  # scale confidence

        # Styled output
        if prediction == "FAKE":
            st.error(f"🔴 This news is likely FAKE")
        else:
            st.success(f"🟢 This news is likely REAL")

        st.info(f"📊 Confidence Score: *{confidence}%*")

        # WordCloud
        st.subheader("🧾 Word Cloud of Title")
        wc = WordCloud(width=800, height=400, background_color='white').generate(title)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Export result
        if st.button("📁 Export this prediction"):
            df = pd.DataFrame([[title, prediction]], columns=["Title", "Prediction"])
            df.to_csv("prediction_output.csv", index=False)
            st.success("✅ Saved as prediction_output.csv")

# =========================
# 🔹 CSV Evaluation Section
# =========================
st.markdown("---")
st.subheader("📂 Evaluate Model with CSV")

uploaded_file = st.file_uploader("Upload a CSV with 'Title' and 'Label' columns", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    test_df = pd.read_csv(uploaded_file)

    if "Title" not in test_df.columns or "Label" not in test_df.columns:
        st.error("❌ CSV must contain 'Title' and 'Label' columns.")
    else:
        # Vectorize and predict
        X_test = vectorizer.transform(test_df["Title"])
        y_true = test_df["Label"]
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        st.success(f"✅ Model Accuracy: {accuracy:.2%}")

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Export predictions
        test_df["Prediction"] = y_pred
        test_df.to_csv("evaluation_results.csv", index=False)
        st.info("📁 Results saved as evaluation_results.csv")

# Footer
st.markdown("---")
st.caption("Built with ❤ using Streamlit and scikit-learn.")

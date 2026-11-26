import streamlit as st
import joblib
import string

# Load the saved model and vectorizer
artifacts = joblib.load("models/sms_spam_detector.pkl")

model = artifacts["model"]
vectorizer = artifacts["vectorizer"]

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩"
)

st.title("📩 SMS Spam Detector")

st.write(
    "Enter an SMS message below to check whether it is spam or not."
)

message = st.text_area("Enter your SMS message:")

if st.button("Check Message"):

    if message.strip():

        cleaned_message = message.lower()

        cleaned_message = cleaned_message.translate(
            str.maketrans('', '', string.punctuation)
        )

        cleaned_message = ' '.join(cleaned_message.split())

        message_vector = vectorizer.transform([cleaned_message])

        prediction = model.predict(message_vector)[0]

        probabilities = model.predict_proba(message_vector)[0]

        confidence = probabilities[prediction] * 100

        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")

        st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter a message.")

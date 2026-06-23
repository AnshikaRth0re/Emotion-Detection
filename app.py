
import streamlit as st
import pickle

# Page Config
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🧠",
    layout="centered"
)

# Load Model and Vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Label Mapping
emotion_labels = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

# Emojis
emotion_emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# Title
st.title("🧠 Emotion Detection System")
st.write("This application predicts the emotion expressed in a given text ")

# Input
text = st.text_area(
    "✍ Enter your text here",
    placeholder="Example: I am feeling very happy today!"
)

# Prediction
if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([text])
        prediction = int(model.predict(vector)[0])

        emotion = emotion_labels.get(prediction, "Unknown")
        emoji = emotion_emoji.get(emotion, "")

        st.success(f"Predicted Emotion: {emotion.upper()} {emoji}")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info("""
Emotion Detection System

Technologies Used:
• Python
• Scikit-Learn
• TF-IDF
• Streamlit
""")



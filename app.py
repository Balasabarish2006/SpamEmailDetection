
import streamlit as st
import pickle
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detection")
st.write("Check whether your message is Spam or Not Spam.")

# Load trained model
model_path = Path(__file__).parent / "spam_model.pkl"

if not model_path.exists():
    st.error("Model not found. Run py train_model.py first.")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Store history
if "history" not in st.session_state:
    st.session_state.history = []

# Message input
message = st.text_area(
    "Enter your email or message:",
    placeholder="Example: Congratulations! You won a free prize...",
    height=150
)

# Predict
if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)

        spam_probability = float(
            probabilities[classes.index("spam")]
        )
        confidence = float(max(probabilities) * 100)

        if prediction == "spam":
            st.error("⚠️ This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")

        st.metric(
            "Spam Probability",
            f"{spam_probability * 100:.1f}%"
        )
        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

        st.progress(spam_probability)

        st.session_state.history.insert(0, {
            "Message": message,
            "Prediction": prediction.upper(),
            "Spam Probability": f"{spam_probability * 100:.1f}%"
        })

# History section
st.divider()
st.subheader("📋 Message History")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No messages checked yet.")

st.caption(
    "Note: Predictions are estimates based on training data, "
    "not a guarantee that a message is safe."
)


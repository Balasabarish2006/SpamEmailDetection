
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample email dataset
data = {
    "message": [
        # Spam messages
        "Win a free prize now",
        "Congratulations you won a lottery",
        "Claim your free gift today",
        "You have won free money",
        "Click here to win a prize",
        "Get cash now",
        "Claim your reward now",
        "Free offer just for you",
        "You are a lucky winner",
        "Win money instantly",
        "Free gift waiting for you",
        "Congratulations claim your prize",
        "Earn money fast click here",
        "Exclusive free offer today",
        "You won a big prize",
        "Claim your lottery money",
        "Get a free bonus now",
        "Urgent claim your reward",
        "Win a free vacation",
        "Click here for free cash",

        # Normal messages
        "Meeting scheduled for tomorrow",
        "Please send the project report",
        "Let's discuss the assignment",
        "Your appointment is confirmed",
        "Please attend the meeting",
        "Hi how are you",
        "Have a nice day",
        "Hello how are you doing",
        "Can we meet tomorrow",
        "Please call me later",
        "I will send the report today",
        "Thank you for your help",
        "The meeting starts at ten",
        "Please complete your homework",
        "Let us discuss the project",
        "See you tomorrow morning",
        "How was your day",
        "I am doing well thank you",
        "Please check your email",
        "Have a good day"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

# Create the machine learning pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train using all available sample messages
model.fit(df["message"], df["label"])

# Save the trained model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Total training messages:", len(df))
print("Spam messages:", (df["label"] == "spam").sum())
print("Normal messages:", (df["label"] == "ham").sum())
print("Model saved as spam_model.pkl")

# Test sample messages
test_messages = [
    "Hi, how are you? Have a nice day.",
    "Congratulations! You won a free prize. Click here now!"
]

for message in test_messages:
    prediction = model.predict([message])[0]
    print("\nMessage:", message)
    print("Prediction:", prediction.upper())

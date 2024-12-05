import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure nltk resources are available
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Function to load and write conversation history to CSV
def update_conversation_log(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Streamlit application
def main():
    st.title("Intents-based Chatbot using NLP")

    # Sidebar Menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message and press Enter to start the conversation.")

        # Input Box for User to Type
        user_input = st.text_input("You:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None)
            
            # Save the conversation to log
            update_conversation_log(user_input, response)

            # Stop the conversation if 'bye' or 'goodbye' is detected
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")

        # Display conversation history from CSV
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except Exception as e:
            st.write("Error loading conversation history:", e)

    elif choice == "About":
        st.write("This is an NLP-powered chatbot built with Python using the Logistic Regression algorithm.")
        st.subheader("Project Overview:")
        st.write("""
        This project uses NLP techniques for intent recognition and responses.
        - **Data**: The chatbot is trained using labeled intents (greetings, FAQs, etc.)
        - **Machine Learning**: We use TF-IDF Vectorizer for text transformation and Logistic Regression for classification.
        - **Interface**: Streamlit is used to create an interactive web interface for the chatbot.
        """)

        st.subheader("About the Dataset:")
        st.write("""
        The dataset contains labeled intents, each associated with example user inputs and responses.
        Examples of intents:
        - Greeting
        - Asking for Help
        - Farewell
        """)
        st.subheader("Conclusion:")
        st.write("""
        This project demonstrates how to build a chatbot using NLP techniques and a simple machine learning model.
        You can extend this project by adding more intents, improving the model, or using deep learning for better performance.
        """)

if __name__ == '__main__':
    main()

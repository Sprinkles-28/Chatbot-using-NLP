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

# Arithmetic operations handler
def handle_arithmetic(command):
    command = command.lower().strip()
    if "add" in command:
        numbers = extract_numbers(command)
        return f"The result of addition is: {sum(numbers)}"
    elif "subtract" in command:
        numbers = extract_numbers(command)
        result = numbers[0] - sum(numbers[1:])
        return f"The result of subtraction is: {result}"
    elif "multiply" in command:
        numbers = extract_numbers(command)
        result = 1
        for num in numbers:
            result *= num
        return f"The result of multiplication is: {result}"
    elif "divide" in command:
        numbers = extract_numbers(command)
        if len(numbers) != 2:
            return "Division requires exactly two numbers."
        if numbers[1] == 0:
            return "Cannot divide by zero."
        return f"The result of division is: {numbers[0] / numbers[1]}"
    else:
        return "Invalid operation. Please use add, subtract, multiply, or divide."

# Extract numbers from the user input
def extract_numbers(command):
    words = command.split()
    numbers = []
    for word in words:
        try:
            numbers.append(float(word))
        except ValueError:
            continue
    if len(numbers) < 1:
        raise ValueError("No numbers found. Please provide numbers in your command.")
    return numbers

# Chatbot function
def chatbot(input_text):
    # Check if the user input contains an arithmetic command
    if any(op in input_text.lower() for op in ["add", "subtract", "multiply", "divide"]):
        try:
            return handle_arithmetic(input_text)
        except ValueError as e:
            return str(e)
    # Use intent-based responses
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Counter for unique keys in Streamlit
counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications. For this project, I've also added basic arithmetic functionalities for calculations.")

if __name__ == '__main__':
    main()

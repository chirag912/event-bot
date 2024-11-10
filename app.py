import subprocess
import sys

# Install spacy and model if not already installed
def install_packages():
    try:
        import spacy
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy

    # Attempt to load the Spacy model, installing it if missing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = install_packages()

import openai
import streamlit as st

# Define the function to analyze sentiment with word-level contributions using GPT-4
def analyze_sentiment_with_words(review, category):
    """
    Analyzes the sentiment of the review with word-level contributions.
    
    Parameters:
        review (str): The text of the review.
        category (str): The category of the review (e.g., Food, Product, Place).
    
    Returns:
        str: Sentiment analysis with word-level contributions.
    """
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    sentiment_analysis = response['choices'][0]['message']['content']
    return sentiment_analysis.strip()

# Function to perform Named Entity Recognition (NER)
def extract_entities(review):
    """
    Extracts named entities from the review text.
    
    Parameters:
        review (str): The text of the review.
    
    Returns:
        list: A list of tuples containing entity text and label.
    """
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI
st.title("Sentiment Analysis and Named Entity Recognition App")
st.write("This app analyzes the sentiment of a review with word-level contributions and identifies named entities in the review text.")

# User inputs
category = st.selectbox("Select the category of the review:", ["Food", "Product", "Place", "Other"])
review = st.text_area("Enter your review:")

# Button to perform analysis
if st.button("Analyze"):
    if review:
        # Perform Sentiment Analysis
        st.subheader("Sentiment Analysis with Word-Level Contributions")
        sentiment_with_contributions = analyze_sentiment_with_words(review, category)
        st.write(sentiment_with_contributions)

        # Perform Named Entity Recognition (NER)
        st.subheader("Named Entity Recognition (NER)")
        entities = extract_entities(review)
        if entities:
            st.write(entities)
        else:
            st.write("No named entities found in the review.")
    else:
        st.error("Please enter a valid review.")

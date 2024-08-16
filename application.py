import streamlit as st
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model
model_filename = 'sentiment_model.pkl'
pipeline = joblib.load(model_filename)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [ps.stem(w) for w in words if w not in stop_words and w.isalpha()]
    return ' '.join(words)

def predict_sentiment(review, rating):
    # Preprocess the review text
    processed_review = preprocess_text(review)
    
    # Combine the processed review and rating into a DataFrame
    input_data = pd.DataFrame({'Processed_Review': [processed_review], 'Rating': [rating]})
    
    # Predict sentiment using the pipeline
    sentiment = pipeline.predict(input_data)[0]
    return sentiment

def main():
    st.title('Sentiment Analysis App')

    st.write("Enter a review and rating to get sentiment prediction:")

    review = st.text_area('Review:')
    rating = st.slider('Rating:', min_value=0, max_value=5, value=3)

    if st.button('Predict Sentiment'):
        if review:
            sentiment = predict_sentiment(review, rating)
            st.write(f'The predicted sentiment is: **{sentiment}**')
        else:
            st.write('Please enter a review.')

if __name__ == '__main__':
    main()

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved vectorizer, model, and label encoder
vectorizer_filename = 'tfidf_vectorizer.pkl'
model_filename = 'logistic_regression_model.pkl'
label_encoder_filename = 'label_encoder.pkl'

vectorizer = joblib.load(vectorizer_filename)
model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [ps.stem(w) for w in words if w not in stop_words and w.isalpha()]
    return ' '.join(words)

def predict_sentiment(review, rating):
    processed_review = preprocess_text(review)
    review_transformed = vectorizer.transform([processed_review]).toarray()
    rating_transformed = np.array([[rating]])
    X_combined = np.hstack((review_transformed, rating_transformed))
    y_pred = model.predict(X_combined)
    sentiment = label_encoder.inverse_transform(y_pred)[0]
    return sentiment

def main():
    st.title('Sentiment Analysis App')
    
    # Create input fields for the review and rating
    review = st.text_area('Enter your review:')
    rating = st.slider('Rate the product (1 to 5):', min_value=1, max_value=5)
    
    if st.button('Predict Sentiment'):
        if review:
            sentiment = predict_sentiment(review, rating)
            st.write(f'The predicted sentiment is: **{sentiment}**')
        else:
            st.write('Please enter a review.')
            
if __name__ == '__main__':
    main()

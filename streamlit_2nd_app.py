import streamlit as st
import joblib
import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model
model_filename = 'sentiment_model_only_text.pkl'
pipeline = joblib.load(model_filename)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    """Preprocess the review text."""
    words = word_tokenize(text.lower())
    words = [ps.stem(w) for w in words if w not in stop_words and w.isalpha()]
    return ' '.join(words)

def predict_sentiment(review):
    """Predict sentiment based on review."""
    # Preprocess the review text
    processed_review = preprocess_text(review)
    
    # Predict sentiment using the pipeline
    sentiment = pipeline.predict([processed_review])[0]
    return sentiment

def main():
    """Main function to run the Streamlit app."""
    st.sidebar.title('Sentiment Analysis App')
    
    with st.sidebar.expander("About", expanded=True):
        st.write("This application uses a machine learning model to predict the sentiment of reviews based on the review text alone.")
    
    st.title('Sentiment Analysis App')
    
    st.write("## Enter Review and Rating")
    st.write("Provide a review and rating. The rating is included for your reference but does not influence the sentiment prediction.")
    
    # Create user input fields
    review = st.text_area('Review:', help='Enter the review text here.')
    rating = st.slider('Rating:', min_value=0, max_value=5, value=3, help='Select the rating from 0 to 5.')
    
    # Add a section for example reviews
    with st.sidebar.expander("Example Reviews", expanded=True):
        st.write("**Example 1:**")
        st.write("Review: 'This product is fantastic!'")
        st.write("Rating: 5")
        st.write("Sentiment: Positive")
        
        st.write("**Example 2:**")
        st.write("Review: 'I had a terrible experience with this service.'")
        st.write("Rating: 1")
        st.write("Sentiment: Negative")
    
    # Display current input values
    st.write("### Current Input Values")
    st.write(f"**Review:** {review}")
    st.write(f"**Rating:** {rating}")
    
    # Button to trigger sentiment prediction
    if st.button('Predict Sentiment'):
        if review:
            sentiment = predict_sentiment(review)
            st.write(f'### The predicted sentiment is: **{sentiment}**')
        else:
            st.write('Please enter a review.')
    
    # Prediction history (for demonstration purposes)
    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button('Add to History'):
        if review:
            sentiment = predict_sentiment(review)
            st.session_state.history.append({
                'Review': review,
                'Rating': rating,  # Include rating in history
                'Sentiment': sentiment
            })

    if st.session_state.history:
        st.write("### Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.write(history_df)
        
        # Add download option for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction History",
            data=csv,
            file_name='prediction_history.csv',
            mime='text/csv'
        )
    
    # Model Information and Instructions
    st.write("### Model Information")
    st.write("This sentiment analysis model considers only the review text to predict sentiment. The review text is preprocessed and used for the prediction. The rating is not used in the prediction but is included for reference.")

if __name__ == '__main__':
    main()
import streamlit as st
from sentiment_analysis import predict_sentiment

# Streamlit App Interface
st.title("Sentiment Analysis App")
st.write("Enter a review below, and the app will return the sentiment using the Hugging Face model.")
st.subheader("Enter the reviews or Feedback")

# Text input for the review
review = st.text_area("Enter the review text below")

if st.button("Analyze Sentiment"):
    if review:
        sentiment = predict_sentiment(review)
        
        if "error" in sentiment:
            # Display an error message if there's an error in sentiment prediction
            st.error(f"Error during prediction: {sentiment['error']}")
        else:
            # Display the sentiment result
            st.write(f"Sentiment: {sentiment.get('Sentiment', 'No sentiment data returned')}")
    else:
        st.warning("Please enter a review to analyze.")

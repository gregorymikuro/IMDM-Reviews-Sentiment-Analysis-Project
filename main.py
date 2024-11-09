import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier

# Load the saved model
MODEL_PATH = 'models/best_catboost_model.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"The model file was not found at {MODEL_PATH}. Please ensure the model is saved correctly.")

with open(MODEL_PATH, 'rb') as model_file:
    best_catboost = pickle.load(model_file)


# Streamlit App
def preprocess_text(text):
    """Simple preprocessing to clean the input text."""
    return text.lower()


def main():
    st.title("Sentiment Analysis Model Deployment")
    st.write("Predict sentiment of movie reviews with the CatBoost model")

    # Select input type
    input_type = st.radio("Select input type for prediction:", ("CSV File", "Single Sentence"))

    # Option 1: Predict from a CSV File
    if input_type == "CSV File":
        uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])
        if uploaded_file is not None:
            # Read uploaded CSV file
            input_df = pd.read_csv(uploaded_file)
            if 'clean_review' in input_df.columns:
                st.write("Predicting sentiment for uploaded CSV...")

                # Vectorize the input using TF-IDF in real-time
                vectorizer_tfidf = TfidfVectorizer(max_features=5000)
                X_input = vectorizer_tfidf.fit_transform(input_df['clean_review'])
                predictions = best_catboost.predict(X_input)
                input_df['sentiment_prediction'] = predictions
                st.write(input_df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=input_df.to_csv(index=False).encode('utf-8'),
                    file_name='sentiment_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error("The CSV must contain a 'clean_review' column for prediction.")

    # Option 2: Predict from a Single Sentence
    elif input_type == "Single Sentence":
        user_input = st.text_area("Enter a movie review for sentiment prediction:")
        if user_input:
            # Preprocess the input and vectorize it
            preprocessed_input = preprocess_text(user_input)
            vectorizer_tfidf = TfidfVectorizer(max_features=5000)
            X_input = vectorizer_tfidf.fit_transform([preprocessed_input])
            prediction = best_catboost.predict(X_input)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"Predicted Sentiment: {sentiment}")


if __name__ == "__main__":
    main()

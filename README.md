# Twitter Sentiment Analysis
This project analyzes the sentiment of Twitter data using a machine learning approach. It provides predictions for whether a given text is positive or negative based on a trained model. The application is built with Python and deployed using Streamlit.

Live demo : https://tweet-tone.streamlit.app/

## Features
- Preprocessing of raw text data (lowercasing, punctuation removal, stopword removal, lemmatization).
- Sentiment analysis using Logistic Regression and Random Forest classifiers.
- Interactive Streamlit interface for user input and sentiment prediction.

## Dataset
The application uses a preprocessed dataset (`Twitter_Data.csv`) that includes a column named `clean_text` for the text and `sentiment` for the labels (positive/negative).

## Model Performance
- **Logistic Regression**: High accuracy with cross-validation.
- **Random Forest**: Alternative model with competitive performance.


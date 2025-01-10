import streamlit as st
import pandas as pd
import re
import nltk
from textblob import Word
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# NLTK veri dosyalarını yükleme
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
except LookupError:
    st.error("Gerekli NLTK veri dosyaları yüklenemedi. Lütfen 'nltk.download()' komutlarını çalıştırın.")
    sw = set()  # Hata durumunda boş set

# Veri yükleme ve temizleme
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Twitter_Data.csv")
    df.rename(columns={'category': 'sentiment'}, inplace=True)
    df['sentiment'] = df['sentiment'].map({1.0: 'pos', 0.0: 'neutral', -1.0: 'neg'})
    df = df.dropna(subset=['sentiment'])
    df['clean_text'] = df['clean_text'].fillna("")
    df['clean_text'] = df['clean_text'].astype(str)
    df['clean_text'] = df['clean_text'].str.lower()
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\d', '', x))
    df['clean_text'] = df['clean_text'].apply(lambda x: " ".join(word for word in x.split() if word not in sw))
    df['clean_text'] = df['clean_text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
    return df

df1 = load_and_clean_data()

# Model eğitimi
train_x, test_x, train_y, test_y = train_test_split(df1['clean_text'], df1['sentiment'], random_state=42)

# TF-IDF vektörizasyonu
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Logistic Regression Model
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Kullanıcı girdisi için tahmin fonksiyonu
def predict_sentiment(user_input, model, vectorizer):
    if not user_input.strip():
        return "Lütfen geçerli bir yorum girin."
    try:
        user_input = user_input.lower()
        user_input = re.sub(r'[^\w\s]', '', user_input)
        user_input = re.sub(r'\d', '', user_input)
        user_input = " ".join([Word(word).lemmatize() for word in user_input.split()])
        user_input = " ".join([word for word in user_input.split() if word not in sw])

        user_input_vector = vectorizer.transform([user_input])
        user_prediction = model.predict(user_input_vector)[0]

        if user_prediction == 'neutral':
            return "Girilen yorumun duygusal tonu nötr."
        return "Girilen yorumun duygusal tonu pozitif." if user_prediction == 'pos' else "Girilen yorumun duygusal tonu negatif."
    except Exception as e:
        return f"Bir hata oluştu: {e}"

# Streamlit arayüzü
st.title("Twitter Duygu Analizi")

user_input = st.text_input("Lütfen bir yorum girin:")
if user_input:
    result = predict_sentiment(user_input, log_model, tf_idf_word_vectorizer)
    st.write(result)

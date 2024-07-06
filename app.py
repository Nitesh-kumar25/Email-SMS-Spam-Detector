import streamlit as st
import pandas as pd
import pickle
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def text_preprocess(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Detector')

input_message = st.text_area('Enter the message/mail')

# preprocessing
transformed_msg = text_preprocess(input_message)

# Vectorization
vectors = tfidf.transform([transformed_msg])

# Predict
result=model.predict(vectors)

if st.button('Predict'):
    if result == 1:
        st.subheader('Spam')
    else:
        st.subheader('Not spam')

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model('roman_urdu_poetry_lstm.h5')

# Tokenizer (To be initialized with dataset)
def train_tokenizer():
    data = pd.read_csv('Roman-Urdu-Poetry.csv', encoding='utf-8')
    lines = data['Poetry'].dropna().str.strip().str.lower().tolist()
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = train_tokenizer()
total_chars = len(tokenizer.word_index) + 1
sequence_length = 40

# Poetry Generation Function
def generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars=100):
    generated = seed_text
    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted)
        output_char = tokenizer.index_word.get(predicted_id, '')
        generated += output_char
        seed_text = seed_text[1:] + output_char
    return generated

# Streamlit UI
st.title("Roman Urdu Poetry Generator 💬")
model = load_trained_model()

seed_text = st.text_input("Enter Seed Text:", "meri dill ki dhadkan")
num_chars = st.slider("Number of Characters to Generate:", 50, 300, 100)

generate_button = st.button("Generate Poetry ✨")

if generate_button:
    with st.spinner("Generating poetry..."):
        poetry = generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars)
    st.subheader("Generated Poetry:")
    st.write(poetry)

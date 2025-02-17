import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model("roman_urdu_poetry_lstm.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set sequence length (make sure it matches training)
sequence_length = 40  

def generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars=100):
    generated = seed_text
    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")
        predicted = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted)
        output_char = tokenizer.index_word.get(predicted_id, "")  # Handle missing index
        generated += output_char
        seed_text = seed_text[1:] + output_char  # Update seed text
    return generated

# Streamlit UI
st.title("Roman Urdu Poetry Generator 📝")
st.write("Enter a starting word or phrase, and the model will generate poetry.")

# User input
seed_text = st.text_input("Enter a seed word or phrase:", "meri dill ki dhadkan")

if st.button("Generate Poetry"):
    poetry = generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars=200)
    st.subheader("Generated Poetry:")
    st.write(poetry)

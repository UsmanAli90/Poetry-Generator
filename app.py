import streamlit as st
import numpy as np
import pickle
import textwrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("roman_urdu_poetry_lstm.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

sequence_length = 40  

def generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars=100, line_length=40, stanza_lines=2):
    generated = seed_text
    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")
        predicted = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted)
        output_char = tokenizer.index_word.get(predicted_id, "") 
        generated += output_char
        seed_text = seed_text[1:] + output_char  


    poetry_lines = textwrap.wrap(generated, width=line_length)  
    formatted_poetry = "\n".join(
        "\n".join(poetry_lines[i : i + stanza_lines]) + "\n" for i in range(0, len(poetry_lines), stanza_lines)
    )

    return formatted_poetry.strip()

st.title("Roman Urdu Poetry Generator 📝")
st.write("Enter a starting word or phrase, and the model will generate poetry.")

seed_text = st.text_input("Enter a seed word or phrase:", "meri dill ki dhadkan")

if st.button("Generate Poetry"):
    poetry = generate_poetry(seed_text, model, tokenizer, sequence_length, num_chars=200)
    st.subheader("Generated Poetry:")
    st.write(poetry)

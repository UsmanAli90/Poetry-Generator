import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/Roman-Urdu-Poetry.csv', encoding='utf-8')
lines = data['Poetry'].dropna().str.strip().str.lower().tolist()  

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(lines)
total_chars = len(tokenizer.word_index) + 1  

sequence_length = 40  
sequences = []
next_chars = []

for line in lines:
    if len(line) > sequence_length:
        for i in range(0, len(line) - sequence_length):
            seq = line[i:i + sequence_length]
            next_char = line[i + sequence_length]
            sequences.append(seq)
            next_chars.append(next_char)

X = tokenizer.texts_to_sequences(sequences)
X = pad_sequences(X, maxlen=sequence_length, padding='pre')
y = tokenizer.texts_to_sequences(next_chars)
y = np.array(y)
y = to_categorical(y, num_classes=total_chars)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(total_chars, 50, input_length=sequence_length))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(total_chars, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128,
    verbose=1
)

model.save('roman_urdu_poetry_lstm.h5')

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

seed = "meri dill ki dhadkan"  
print(generate_poetry(seed, model, tokenizer, sequence_length, num_chars=200))

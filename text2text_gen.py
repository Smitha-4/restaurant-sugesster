import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical

# Load data and clean text (improve cleaning steps here)
df = pd.read_csv('cleaned_file')
def combine_text(row):
  cuisine = row["cuisines"].lower()
  menu_item = row["menu items"].lower()
  return cuisine + ", " + menu_item  # Add a delimiter

df["combined_text"] = df.apply(combine_text, axis=1)

# Create character dictionary (potentially save and load from a file)
chars = sorted(list(set(df['combined_text'])))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_vocab = len(chars)

seq_length = 200
x_data = []
y_data = []
for i in range(0, len(df['combined_text']) - seq_length, 1):
    seq_in = df['combined_text'][i:i + seq_length]
    seq_out = df['combined_text'][i + seq_length]
    x_data.append([char_to_int[c] for c in seq_in])
    y_data.append(char_to_int[seq_out])

n_patterns = len(x_data)

x = np.reshape(x_data, (n_patterns, seq_length, 1))
x = x / float(n_vocab)
y = to_categorical(y_data)

# Define model architecture (consider exploring GRUs or Transformers)
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))  # Enable return sequences for multi-step prediction
model.add(Dropout(0.2))
model.add(LSTM(128))  # Experiment with additional layers or units
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x, y, epochs=20, batch_size=128)

# Save the entire model
model.save('cuisines_to_menu.h5')

# Load the model
loaded_model = tf.keras.models.load_model('cuisines_to_menu.h5')

start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Seed:")
print("'", ''.join([int_to_char[value] for value in pattern]), "'")

# Generate text with multi-step prediction
for i in range(1000):
    prediction = loaded_model.predict(np.expand_dims(pattern, axis=0))[0]  # Predict multiple characters at once
    index = np.argmax(prediction)
    result = int_to_char

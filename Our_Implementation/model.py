import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

# Funzione per caricare il dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['pairs']), np.array(data['labels'])

# Carica il dataset
pairs, labels = load_dataset('dataset.pickle')

# Parametri del modello
sequence_length = pairs[0][0].shape[0]  # Lunghezza della sequenza dai dati
feature_count = pairs[0][0].shape[1]    # Numero di feature per pacchetto
lstm_units = 64                         # Numero di unità LSTM
learning_rate = 0.001                   # Learning rate

# Definisci la sottorete LSTM per estrarre caratteristiche dalle sequenze
def build_siamese_branch(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=False)(input_layer)
    x = Dense(32, activation="relu")(x)
    return Model(input_layer, x)

# Input per le sequenze entry e exit
input_shape = (sequence_length, feature_count)
entry_input = Input(shape=input_shape, name="entry_input")
exit_input = Input(shape=input_shape, name="exit_input")

# Costruisci due rami identici per le sequenze entry e exit
siamese_branch = build_siamese_branch(input_shape)
entry_embedding = siamese_branch(entry_input)
exit_embedding = siamese_branch(exit_input)

# Calcola la distanza L1 tra le due rappresentazioni
def l1_distance(tensors):
    x, y = tensors
    return tf.abs(x - y)

distance = Lambda(l1_distance)([entry_embedding, exit_embedding])
output = Dense(1, activation="sigmoid")(distance)

# Definisci il modello completo
siamese_model = Model(inputs=[entry_input, exit_input], outputs=output)

# Compila il modello
siamese_model.compile(optimizer=Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

# Prepara i dati di input
entry_sequences = np.array([pair[0] for pair in pairs])
exit_sequences = np.array([pair[1] for pair in pairs])

# Allena il modello
siamese_model.fit([entry_sequences, exit_sequences], labels, batch_size=16, epochs=10, validation_split=0.2)

# Salva il modello
siamese_model.save("siamese_model.h5")
print("Modello siamese salvato come 'siamese_model.h5'.")

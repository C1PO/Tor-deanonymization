import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Concatenate

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

# Funzione per caricare il dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['pairs']), np.array(data['labels'])

def model_cnn_lstm(input_shape, dropout_keep_prob):
    inputs = tf.keras.Input(shape=input_shape)
    last_layer = inputs

    # Definizione dei layer CNN
    CNN_LAYERS = [[2, 2, 1, 32, 2],  # [filter_height, filter_width, in_channels, out_channels, pool_size]
                  [2, 2, 32, 64, 2]]
    
    for filter_height, filter_width, in_channels, out_channels, pool_size in CNN_LAYERS:
        conv = layers.Conv2D(filters=out_channels,
                             kernel_size=(filter_height, filter_width),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
        
        pool = layers.MaxPooling2D(pool_size=(1, pool_size), strides=(1, 1), padding='valid')(conv)
        last_layer = pool

    # Flatten the output after the last CNN layer
    last_layer = Flatten()(last_layer)

    # Fully connected layers
    flat_layers_after = [256, 128, 64, 1]  # Struttura delle dimensioni dei layer fully connected
    
    for l in range(len(flat_layers_after) - 1):
        output_size = flat_layers_after[l + 1]
        
        last_layer = layers.Dense(units=output_size,
                                  activation='relu' if l < len(flat_layers_after) - 2 else None,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
        
        if l < len(flat_layers_after) - 2:
            last_layer = layers.Dropout(rate=1 - dropout_keep_prob)(last_layer)

    outputs = layers.Activation('sigmoid')(last_layer)
    
    # Modello CNN
    cnn_model = Model(inputs=inputs, outputs=outputs)
    
    # Aggiunta del layer LSTM
    lstm_input_shape = (13, 5)  # Dimensione temporale, dimensione delle features
    lstm_input = layers.Input(shape=lstm_input_shape)
    lstm_output = layers.LSTM(64)(lstm_input)
    
    # Concatenazione del risultato della CNN e LSTM
    concatenated = layers.Concatenate()([cnn_model.output, lstm_output])
    final_output = layers.Dense(1, activation='sigmoid')(concatenated)
    
    # Modello finale con CNN e LSTM
    final_model = Model(inputs=[cnn_model.input, lstm_input], outputs=final_output)
    
    return final_model

# Carica il dataset
pairs, labels = load_dataset('dataset.pickle')
pairs = pairs.astype(np.float32)
labels = labels.astype(np.float32)

# Parametri del modello
sequence_length = pairs[0][0].shape[0]
feature_count = pairs[0][0].shape[1]
lstm_units = 64
learning_rate = 0.001
dropout_keep_prob = 0.5  # Imposta il tasso di dropout

# Costruisci il modello CNN + LSTM
input_shape = (sequence_length, feature_count, 1)  # Aggiungi una dimensione per il canale
model = model_cnn_lstm(input_shape, dropout_keep_prob)

# Compila il modello
model.compile(optimizer=Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

# Prepara i dati di input
entry_sequences = np.array([pair[0] for pair in pairs])
exit_sequences = np.array([pair[1] for pair in pairs])

# Allena il modello
model.fit([entry_sequences, exit_sequences], labels, batch_size=16, epochs=10)

# Salva il modello
model.save("siamese_model.h5")
print("Modello siamese salvato come 'siamese_model.h5'.")
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


# Verifica se ci sono GPU disponibili
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Numero di GPU disponibili: {len(gpus)}")
    for gpu in gpus:
        print("Nome GPU:", gpu.name)
        
        # Opzionalmente, imposta la GPU per limitare la memoria utilizzata
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("Nessuna GPU trovata. Assicurati che i driver siano correttamente installati.")



# Funzione per caricare il dataset
def load_dataset(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return np.array(data['pairs']), np.array(data['labels'])

def siamese_cnn(input_shape, dropout_keep_prob):
    # Definisci il modello CNN per ciascun ramo (entry e exit)
    def cnn_branch(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        last_layer = inputs

        # Convolution Layer 1
        conv1 = layers.Conv2D(filters=2000,
                              kernel_size=(2, 3),  # adattato per il tuo input
                              strides=(2, 1),
                              padding='valid',
                              activation='relu',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
        pool1 = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 1), padding='valid')(conv1)

        # Convolution Layer 2
        conv2 = layers.Conv2D(filters=1000,
                              kernel_size=(2, 2),  # adattato per il tuo input
                              strides=(2, 1),
                              padding='valid',
                              activation='relu',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(pool1)
        # Riduciamo il pool size a (1, 1) per evitare dimensioni negative
        pool2 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv2)

        # Flatten the output after the last CNN layer
        last_layer = layers.Flatten()(pool2)

        return inputs, last_layer

    # Crea due rami CNN, uno per entry e uno per exit
    input_entry, output_entry = cnn_branch(input_shape)
    input_exit, output_exit = cnn_branch(input_shape)

    # Concatenazione delle uscite dei due rami
    concatenated = layers.Concatenate()([output_entry, output_exit])

    # Fully connected layers
    last_layer = layers.Dense(units=3000,
                              activation='relu',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concatenated)
    last_layer = layers.Dropout(rate=1 - dropout_keep_prob)(last_layer)

    last_layer = layers.Dense(units=800,
                              activation='relu',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
    last_layer = layers.Dropout(rate=1 - dropout_keep_prob)(last_layer)

    last_layer = layers.Dense(units=100,
                              activation='relu',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
    last_layer = layers.Dropout(rate=1 - dropout_keep_prob)(last_layer)

    # Output layer (final probability)
    outputs = layers.Dense(1, activation='sigmoid')(last_layer)

    # Crea il modello con i due input e l'output concatenato
    model = Model(inputs=[input_entry, input_exit], outputs=outputs)
    
    return model

# Carica il dataset
pairs, labels = load_dataset('dataset.pickle')
pairs = pairs.astype(np.float32)
labels = labels.astype(np.float32)

def shuffle_data(pairs, labels):
    indices = np.arange(len(pairs))  # Crea una lista di indici
    np.random.shuffle(indices)       # Mescola gli indici

    # Applica lo shuffle ai dati e alle etichette
    pairs = pairs[indices]
    labels = labels[indices]
    return pairs, labels

pairs, labels = shuffle_data(pairs, labels) 

# Suddivisione del dataset in train, validation e test
pairs_train, pairs_temp, labels_train, labels_temp = train_test_split(pairs, labels, test_size=0.3, random_state=42)
pairs_val, pairs_test, labels_val, labels_test = train_test_split(pairs_temp, labels_temp, test_size=2/3, random_state=42)

# Definizione delle forme di input e dei parametri del modello
sequence_length = pairs[0][0].shape[0]
feature_count = pairs[0][0].shape[1]

input_shape = (sequence_length, feature_count, 1)
lstm_input_shape = (18, 5)
dropout_rate = 0.5
learning_rate = 0.001

# Costruzione del modello CNN-LSTM
model = siamese_cnn(input_shape, dropout_rate)
opt = Adam(learning_rate=learning_rate, clipvalue=1.0)  # Limita i gradienti a un valore massimo
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


# Prepara i dati di input
entry_sequences_train = np.array([pair[0] for pair in pairs_train])
exit_sequences_train = np.array([pair[1] for pair in pairs_train])
entry_sequences_val = np.array([pair[0] for pair in pairs_val])
exit_sequences_val = np.array([pair[1] for pair in pairs_val])
entry_sequences_test = np.array([pair[0] for pair in pairs_test])
exit_sequences_test = np.array([pair[1] for pair in pairs_test])

# Allena il modello con riduzione del learning rate on plateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
model.fit([entry_sequences_train, exit_sequences_train], labels_train,
          validation_data=([entry_sequences_val, exit_sequences_val], labels_val),
          batch_size=16, epochs=10, callbacks=[lr_scheduler])

# Valutazione del modello sul set di test
test_loss, test_accuracy = model.evaluate([entry_sequences_test, exit_sequences_test], labels_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Salvataggio del modello
model.save("siamese_model.h5")
print("Modello siamese salvato come 'siamese_model.h5'.")

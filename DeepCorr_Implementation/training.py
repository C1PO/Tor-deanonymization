from model import model_cnn
from import_dataset import *
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def train_model():
    l2s, labels, l2s_test, labels_test = generate_data()

    input_shape = (8, 500,1) 
    dropout_keep_prob = 0.5
    batch_size = 256
    epochs = 200

    model = model_cnn(input_shape, dropout_keep_prob)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',  # Adatto per classificazione binaria
                metrics=['accuracy'])

    model.summary()

    '''
    history = model.fit(l2s, labels, 
                        batch_size=batch_size, 
                        epochs=epochs)

    acc = history.history['accuracy'][-1]
    if acc > 0.8:
        model.save('model.h5')
        print("Modello salvato!")
    '''
    print("faccio il train")
    model.save('model.h5')
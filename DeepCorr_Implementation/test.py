import tensorflow as tf
from import_dataset import *


def test_model(l2s_test, labels_test):
    model = tf.keras.models.load_model('model.h5')

    model.summary()

    test_loss, test_acc = model.evaluate(l2s_test, labels_test, verbose=2)

    print(f"Accuratezza del modello sui dati di test: {test_acc * 100:.2f}%")

    predizioni = model.predict(nuovi_dati)
    print(predizioni)

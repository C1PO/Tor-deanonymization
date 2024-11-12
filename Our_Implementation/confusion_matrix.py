import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Funzione per caricare il dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['pairs']), np.array(data['labels'])

# Carica il dataset e dividi i dati di test
pairs, labels = load_dataset('dataset.pickle')
pairs = pairs.astype(np.float32)
labels = labels.astype(np.float32)

# Dividi il dataset in train, validation e test (come avevi fatto in origine)
from sklearn.model_selection import train_test_split
pairs_train, pairs_temp, labels_train, labels_temp = train_test_split(pairs, labels, test_size=0.3, random_state=42)
pairs_val, pairs_test, labels_val, labels_test = train_test_split(pairs_temp, labels_temp, test_size=2/3, random_state=42)

# Prepara i dati di test
entry_sequences_test = np.array([pair[0] for pair in pairs_test])
exit_sequences_test = np.array([pair[1] for pair in pairs_test])

# Carica il modello salvato
model = tf.keras.models.load_model("siamese_model.h5")

# Fai previsioni sui dati di test
predictions = model.predict([entry_sequences_test, exit_sequences_test])
predicted_labels = (predictions > 0.5).astype(int)

# Calcola e visualizza la confusion matrix
cm = confusion_matrix(labels_test, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non Correlato", "Correlato"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

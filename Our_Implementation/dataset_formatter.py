import os
import itertools
import numpy as np
import pandas as pd
import pickle

# Parametri di configurazione
feature_cols = ['Size', 'Sequence Number', 'Acknowledgment Number', 'TSval', 'TSecr']

# Funzione di caricamento delle sequenze dai CSV
def load_sequence(csv_path, sequence_length, feature_cols):
    df = pd.read_csv(csv_path)
    df = df[feature_cols].fillna(0)  # Sostituisci i NaN se presenti
    sequence = df.values[:sequence_length]  # Trunca o riempi le sequenze
    if len(sequence) < sequence_length:
        # Padding con zeri se la sequenza è troppo corta
        padding = np.zeros((sequence_length - len(sequence), len(feature_cols)))
        sequence = np.vstack([sequence, padding])
    return sequence

# Trova tutti i file CSV nelle cartelle `entry` e `exit`
entry_files = sorted([os.path.join('entry', f) for f in os.listdir('entry') if f.endswith('.csv')])
exit_files = sorted([os.path.join('exit', f) for f in os.listdir('exit') if f.endswith('.csv')])

# Calcola la sequence_length come la lunghezza del file più piccolo - 1
min_sequence_length = min([pd.read_csv(f).shape[0] for f in entry_files + exit_files])
print("Sequence length impostata a:", min_sequence_length)

# Prepara i dati
entry_sequences = [load_sequence(f, min_sequence_length, feature_cols) for f in entry_files]
exit_sequences = [load_sequence(f, min_sequence_length, feature_cols) for f in exit_files]

# Genera le coppie di sequenze e le rispettive etichette
pairs = []
csv_pairs=[]
csv_uno=[]
csv_zero=[]
labels = []
for i in range (len(entry_files)):
    csv_pairs.append((entry_files[i],exit_files[i]))
    csv_uno.append((entry_files[i],exit_files[i]))
for (pippo,poppo) in itertools.product(entry_files, exit_files):
    if (pippo,poppo) not in csv_pairs:
        csv_pairs.append((pippo,poppo))
        csv_zero.append((pippo,poppo))
entry_zero = [load_sequence(f[0], min_sequence_length, feature_cols) for f in csv_zero]
exit_zero = [load_sequence(f[1], min_sequence_length, feature_cols) for f in csv_zero]


for i in range(len(entry_sequences)):
    # Coppie correlate (etichetta 1)
    pairs.append((entry_sequences[i], exit_sequences[i]))
    labels.append(1)

for i in range(len(entry_zero)):
    # Coppie correlate (etichetta 1)
    pairs.append((entry_zero[i], exit_zero[i]))
    labels.append(0)
# Converti le coppie e le etichette in un dizionario e salva in un file .pickle
dataset = {
    'pairs': pairs,
    'labels': labels
}
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("Dataset salvato come 'dataset.pickle'.")

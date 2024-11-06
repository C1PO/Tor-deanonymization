import pickle

# Carica il dataset dal file .pickle
with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

# Stampa le dimensioni delle coppie e delle etichette
print("Numero totale di coppie:", len(dataset['pairs']))
print("Numero totale di etichette:", len(dataset['labels']))


# Stampa alcune coppie e le rispettive etichette
for i in range(len(dataset['pairs'])): 
    entry_seq, exit_seq = dataset['pairs'][i]
    label = dataset['labels'][i]
    
    '''print(f"\nCoppia {i+1}:")
    print("Entry Sequence:")
    print(entry_seq)
    print("Exit Sequence:")
    print(exit_seq)'''
    print("Etichetta:", label)
    print(i)
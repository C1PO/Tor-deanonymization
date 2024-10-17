import pickle
import numpy as np
import tqdm
from model import *

all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
         '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
        '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
dataset=[]

for name in all_runs:
    with open(f'./data/{name}_tordata500.pickle', 'rb') as file:
        dataset+= pickle.load(file)
input_shape = (8, 500,1) 
dropout_keep_prob = 0.5
model = model_cnn(input_shape, dropout_keep_prob)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',  # Adatto per classificazione binaria
            metrics=['accuracy'])

model.summary()
len_tr=len(dataset)
train_ratio=0.7
rr= list(range(len(dataset)))
np.random.shuffle(rr)

train_index=rr[:int(len_tr*train_ratio)]



flow_size=500
negative_samples=500

all_samples = len(train_index)

labels = np.zeros((all_samples * (negative_samples + 1), 1))
l2s = np.zeros((all_samples * (negative_samples + 1), 8, flow_size, 1))


portion_size = int(all_samples *0.1)
for start in range(0, all_samples, portion_size):
    end = min(start + portion_size, all_samples)

    index = 0
    train_partial_index=train_index[start:end]
    random_ordering=[]+train_partial_index


    for i in tqdm.tqdm(train_partial_index):
        # Fill in the positive sample
        l2s[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0
        l2s[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0
        
        labels[index,0]=1
        index += 1
        
        # Generate negative samples
        np.random.shuffle(random_ordering)
        m = 0
        
        for idx in random_ordering:
            if  m > (negative_samples-1):
                break
            if idx == i:
                continue
            m += 1
            l2s[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0
            l2s[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0
            
            labels[index, 0] = 0  # Negative sample
            index += 1

    batch_size = 8
    epochs = 2
    '''
    history = model.fit(l2s, labels, 
                        batch_size=batch_size, 
                        epochs=epochs)
    '''
print("fatto")
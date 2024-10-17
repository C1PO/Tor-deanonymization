import pickle
import numpy as np


all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
         '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
        '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
flow_size=500
dataset=[]

for name in all_runs:
    with open(f'./data/{name}_tordata500.pickle', 'rb') as file:
        dataset+= pickle.load(file)


len_tr=len(dataset)
train_ratio=float(len_tr-600)/float(len_tr)
rr= list(range(len(dataset)))
np.random.shuffle(rr)

train_index=rr[:int(len_tr*train_ratio)]
test_index= rr[int(len_tr*train_ratio):]
pickle.dump(test_index,open('test_index300.pickle','wb'))

negative_samples=199

import numpy as np
import tqdm

def generate_data():
    global dataset
    global train_index
    global test_index
    global flow_size
    global negative_samples

    all_samples = len(train_index)
    
    labels = np.zeros((all_samples * (negative_samples + 1), 1))
    l2s = np.zeros((all_samples * (negative_samples + 1), 8, flow_size, 1))

    index = 0
    random_ordering=[]+train_index

    for i in tqdm.tqdm(train_index):
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
            if idx == i or m >= (negative_samples-1):
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

    # Prepare test data
    index_hard=0
    num_hard_test = 0
    l2s_test = np.zeros((len(test_index) * (negative_samples + 1), 8, flow_size, 1))
    labels_test = np.zeros((len(test_index) * (negative_samples + 1)))
    l2s_test_hard=np.zeros((num_hard_test*num_hard_test,2,flow_size,1))

    index = 0
    random_test=[]+test_index

    for i in tqdm.tqdm(test_index):

        m = 0
        
        # Generate negative samples for test data
        np.random.shuffle(random_test)
        for idx in random_test:
            if idx == i or m > (negative_samples-1):
                continue
            
            m += 1
            
            l2s_test[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
            l2s_test[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            l2s_test[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            l2s_test[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0
            l2s_test[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
            l2s_test[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            l2s_test[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            l2s_test[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0
            
            labels_test[index] = 0  # Negative sample
            index += 1

        # Add the positive sample for testing
        l2s_test[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s_test[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s_test[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s_test[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0
        l2s_test[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s_test[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s_test[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s_test[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0
        
        labels_test[index] = 1  # Positive sample
        index += 1

    return l2s, labels, l2s_test, labels_test
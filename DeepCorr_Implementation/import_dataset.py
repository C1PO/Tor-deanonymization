import numpy as np
import pickle
import tqdm
from sklearn.model_selection import train_test_split


def generate_indices(dataset_size, test_size=0.2, random_state=42):
    indices = np.arange(dataset_size)
    train_index, test_index = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_index, test_index

def load_dataset(name):
    dataset = []
    file_path = f'./data/{name}_tordata500.pickle'
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def generate_train_data(dataset, train_index, flow_size=500, negative_samples=199):
    all_samples = len(train_index)
    
    labels = np.zeros((all_samples * (negative_samples + 1), 1))
    l2s = np.zeros((all_samples * (negative_samples + 1), 8, flow_size, 1))

    index = 0
    random_ordering = train_index.copy()

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
        
        labels[index, 0] = 1
        index += 1
        
        # Generate negative samples
        np.random.shuffle(random_ordering)
        m = 0
        
        for idx in random_ordering:
            if idx == i or m >= negative_samples:
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
    
    return l2s, labels

def generate_test_data(dataset, test_index, flow_size=500, negative_samples=199):
    l2s_test = np.zeros((len(test_index) * (negative_samples + 1), 8, flow_size, 1))
    labels_test = np.zeros((len(test_index) * (negative_samples + 1)))

    index = 0
    random_test = test_index.copy()

    for i in tqdm.tqdm(test_index):
        m = 0
        
        # Generate negative samples for test data
        np.random.shuffle(random_test)
        for idx in random_test:
            if idx == i or m >= negative_samples:
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
    return l2s_test, labels_test





from import_dataset import *
from training import *
from test import *


all_runs={'8872':'192.168.122.117'}
flow_size=500
dataset_size = 1000
dataset = []
for name in all_runs:
    dataset += load_dataset(name)
train_index, test_index = generate_indices(dataset_size)
l2s, labels = generate_train_data(dataset,train_index)
l2s_test, labels_test = generate_test_data(dataset.test_index)
train_model(l2s, labels)
test_model(l2s_test, labels_test)

'''
,'8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
         '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
        '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'
'''
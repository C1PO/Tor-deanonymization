from training import *
from test import *
from import_dataset import *

l2s, labels= generate_train_data()
train_model(l2s, labels)

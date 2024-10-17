from training import *
from test import *
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Imposta TensorFlow per utilizzare solo la prima GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        #tf.config.experimental.set_memory_growth(gpus[0], True)  # Opzionale, limita l'uso della memoria
    except RuntimeError as e:
        print(e)


train_model()
#test_model()
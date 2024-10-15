import tensorflow as tf
from tensorflow.keras import layers

def model_cnn(input_shape, dropout_keep_prob):
    inputs = tf.keras.Input(shape=input_shape) #differenza
    last_layer = inputs

    # Definizione dei layer CNN come nella lista CNN_LAYERS
    CNN_LAYERS = [[2, 20, 1, 2000, 5], [4, 10, 2000, 800, 3]]
    
    for cnn_size, layer_params in enumerate(CNN_LAYERS):
        filter_height, filter_width, in_channels, out_channels, pool_size = layer_params
        
        # Convolutional Layer
        conv = layers.Conv2D(filters=out_channels,
                             kernel_size=(filter_height, filter_width),
                             strides=(2, 2),
                             padding='valid',
                             activation='relu',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
        
        # Max Pooling
        pool = layers.MaxPooling2D(pool_size=(1, pool_size), strides=(1, 1), padding='valid')(conv)
        last_layer = pool
    
    # Flatten the output after the last CNN layer
    last_layer = layers.Flatten()(last_layer)
    
    # Definire i layer fully connected come in flat_layers_after
    flat_layers_after = [49600, 3000, 800, 100, 1]  # Struttura delle dimensioni dei layer fully connected
    
    for l in range(len(flat_layers_after) - 1):
        output_size = flat_layers_after[l + 1]
        
        # Dense Layer
        last_layer = layers.Dense(units=output_size,
                                  activation='relu' if l < len(flat_layers_after) - 2 else None,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))(last_layer)
        
        # Apply dropout to all fully connected layers except the last one
        if l < len(flat_layers_after) - 2:
            last_layer = layers.Dropout(rate=1 - dropout_keep_prob)(last_layer)
    
    # Output layer (final probability)
    outputs = layers.Activation('sigmoid')(last_layer)
    
    # Creare il modello
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model



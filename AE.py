"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper function

@author Florent Forest
@version 2.0
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose, UpSampling2D


def mlp_autoencoder(encoder_dims, num_dense=2, act='relu', init='glorot_uniform', decoder_type='transpose'):
    """
    Fully connected symmetric autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """

    # hyper parameters
    stamp_sidelength = int(np.sqrt(encoder_dims[0])) # input dim
    cnn_neurons = encoder_dims[1:-num_dense] # CNN neurons number
    dense_neurons = encoder_dims[-num_dense:] # dense layers neurons number


    # model architect
    x = Input(shape=(stamp_sidelength, stamp_sidelength, 1), name='input') # input layer
    input_img = x 
    j = 0
    # add encoder part
    for n in cnn_neurons:
        x = Conv2D(n, (3, 3), activation=act, kernel_initializer=init, padding='same', name='encoder_%d' % j)(x) ; j+=1 # add CNN layer
        x = MaxPooling2D((2, 2), name='encoder_%d' % j)(x)  ; j+=1 # add max pooling layer
    x = Flatten(name='encoder_%d' % j)(x)  ; j+=1 # add layer to flatten Conv2D (proceed to Dense layer)

    for n in dense_neurons[:-1]:
        x = Dense(n, activation=act, kernel_initializer=init, name='encoder_%d' % j)(x) ; j+=1 # add Dense hidden layer
    x = Dense(dense_neurons[-1], kernel_initializer=init, name='encoder_%d' % j)(x) # add Dense hidden layer (latent representation extract from here)
    # create encoder model
    encoder = Model(inputs=input_img, outputs=x, name='encoder')

    if decoder_type == 'upsampling':
        # add decoder part
        k = len(dense_neurons) + 1 + 2*len(cnn_neurons) + 1 - 1 # for indexing the decoder layers
        for j in range(2, len(dense_neurons)+1):
            x = Dense(dense_neurons[-j], activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1 # add Dense layer
        x = Dense(encoder.get_layer(index=-4).output_shape[1]**2*cnn_neurons[-1], activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1 # analogy of flatten layer
        x = Reshape((encoder.get_layer(index=-4).output_shape[1], encoder.get_layer(index=-4).output_shape[1], cnn_neurons[-1]), name='decoder_%d' % k)(x) ; k-=1 # reshape from flatten to Conv2D

        for i in range(1, len(cnn_neurons)+1):
            x = Conv2D(cnn_neurons[-i], (3, 3), padding='same', activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1
            x = UpSampling2D((2, 2), name='decoder_%d' % k)(x) ; k-= 1
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer=init, name='decoder_%d' % k)(x)

        k = len(dense_neurons) + 1 + 2*len(cnn_neurons) + 1 - 1
        # create AE model
        autoencoder = Model(inputs=input_img, outputs=x, name='AE')
        # Create input for decoder model
        encoded_input = Input(shape=(dense_neurons[-1],))
        decoded = encoded_input
        for i in range(k, -1, -1):
            decoded = autoencoder.get_layer('decoder_%d' % i)(decoded) # add layer from AE model in reverse order
        # create decoder model
        decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    elif decoder_type == 'transpose':
        # add decoder part
        k = len(encoder_dims) # for indexing the decoder layers
        for j in range(2, len(dense_neurons)+1):
            x = Dense(dense_neurons[-j], activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1 # add Dense layer
        x = Dense(encoder.get_layer(index=-4).output_shape[1]**2*cnn_neurons[-1], activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1 # analogy of flatten layer
        x = Reshape((encoder.get_layer(index=-4).output_shape[1], encoder.get_layer(index=-4).output_shape[1], cnn_neurons[-1]), name='decoder_%d' % k)(x) ; k-=1 # reshape from flatten to Conv2D
        
        for i in range(1, len(cnn_neurons)+1):
            x = Conv2DTranspose(cnn_neurons[-i], (3, 3), strides=(2, 2), padding='same', activation=act, kernel_initializer=init, name='decoder_%d' % k)(x) ; k-=1
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer=init, name='decoder_%d' % k)(x)

        k = len(encoder_dims)
        # create AE model
        autoencoder = Model(inputs=input_img, outputs=x, name='AE')
        # Create input for decoder model
        encoded_input = Input(shape=(dense_neurons[-1],))
        decoded = encoded_input
        for i in range(k, -1, -1):
            decoded = autoencoder.get_layer('decoder_%d' % i)(decoded) # add layer from AE model in reverse order
        # create decoder model
        decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder

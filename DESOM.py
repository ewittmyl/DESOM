"""
Implementation of the Deep Embedded Self-Organizing Map model
Main file

@author Florent Forest
@version 2.0
"""

# Utilities
import os
import csv
import argparse
from time import time
import matplotlib.pyplot as plt
import shutil
import sys

# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model
from keras.utils.vis_utils import plot_model

# Dataset helper function
from datasets import load_data

# DESOM components
from SOM import SOMLayer
from AE import mlp_autoencoder
from metrics import *


def som_loss(weights, distances):
    """
    SOM loss

    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))


def kmeans_loss(y_pred, distances):
    """
    k-means reconstruction loss

    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])


class DESOM:
    """
    Deep Embedded Self-Organizing Map (DESOM) model

    # Example
        ```
        desom = DESOM(encoder_dims=[784, 500, 500, 2000, 10], map_size=(10,10))
        ```

    # Arguments
        encoder_dims: list of numbers of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer (latent dim)
        map_size: tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1]
    """

    def __init__(self, encoder_dims, num_dense, map_size):
        self.encoder_dims = encoder_dims
        self.num_dense = num_dense
        self.input_dim = self.encoder_dims[0]
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.pretrained = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model = None
        self.AE_frozen = False
    
    def initialize(self, model_type, ae_act='relu', ae_init='glorot_uniform', ae_freeze=False, decoder_type='transpose'):
        """
        Create DESOM model

        # Arguments
            ae_act: activation for AE intermediate layers
            ae_init: initialization of AE layers
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = mlp_autoencoder(self.encoder_dims, self.num_dense, ae_act, ae_init, decoder_type)
        if ae_freeze: # freeze AE
            self.AE_frozen = True
            self.autoencoder.trainable = False
            self.encoder.trainable = False
            self.decoder.trainable = False
        
        self.model_type = model_type

        if model_type == 'desom':
            som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
            # Create DESOM model
            self.model = Model(inputs=self.autoencoder.input,
                            outputs=[self.autoencoder.output, som_layer])
        elif model_type == 'ae':
            # Create DESOM model
            self.model = Model(inputs=self.autoencoder.input,
                                outputs=[self.autoencoder.output])
            
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, optimizer, gamma=0.0075):
        """
        Compile DESOM model

        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        if self.model_type == 'desom':
            if self.AE_frozen:
                self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                                loss_weights=[0, 1],
                                optimizer=optimizer)
            else:
                self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                                loss_weights=[1, gamma],
                                optimizer=optimizer)
        elif self.model_type == 'ae':
            self.model.compile(loss={'decoder_0': 'mse'},
							loss_weights=[1],
							optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DESOM model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def init_som_weights(self, X):
        """
        Initialize with a sample w/o remplacement of encoded data points.

        # Arguments
            X: numpy array containing training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)
    
    def decode(self, x):
        """
        Decoding function. Decodes encoded features from latent space

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer

        # Arguments
            x: data point
        # Return
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)

    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
        
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        # Return
            pairwise distance matrix (map_dist[i,k] is the Manhattan distance on the map between assigned cell of data point i and cell k)
        """
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels) // self.map_size[1]
        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        return d_row + d_col

    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function (gaussian neighborhood)

        # Arguments
            x: distance on the map
            T: temperature parameter
        # Return
            neighborhood weight
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)
    
    def fit(self, X_train,
            X_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
            save_dir='results/tmp'):
        """
        Training procedure

        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           iterations: number of training iterations
           som_iterations: number of iterations where SOM neighborhood is decreased
           eval_interval: evaluate metrics on training/validation batch every eval_interval iterations
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           Tmax: initial temperature parameter
           Tmin: final temperature parameter
           decay: type of temperature decay ('exponential' or 'linear')
           save_dir: path to existing directory where weights and logs are saved
        """
        
        save_interval = X_train.shape[0] // batch_size * save_epochs # save every save_epochs epochs
        print('Save interval:', save_interval)

        # Logging file
        logfile = open(save_dir + '/{}_log.csv'.format(self.model_type), 'w')
        if self.model_type == 'desom':
            if self.AE_frozen:
                fieldnames = ['iter', 'T', 'Lsom', 'Lkm', 'Ltop', 'quantization_err', 'topographic_err', 'latent_quantization_err', 'latent_topographic_err']
                if X_val is not None:
                    fieldnames += ['L_val', 'Lr_val', 'Lsom_val', 'Lkm_val', 'Ltop_val', 'quantization_err_val', 'topographic_err_val', 'latent_quantization_err_val', 'latent_topographic_err_val']
            else:
                fieldnames = ['iter', 'T', 'L', 'Lr', 'Lsom', 'Lkm', 'Ltop', 'quantization_err', 'topographic_err', 'latent_quantization_err', 'latent_topographic_err']
                if X_val is not None:
                    fieldnames += ['L_val', 'Lr_val', 'Lsom_val', 'Lkm_val', 'Ltop_val', 'quantization_err_val', 'topographic_err_val', 'latent_quantization_err_val', 'latent_topographic_err_val']
        elif self.model_type == 'ae':
            fieldnames = ['iter', 'L']
            if X_val is not None:
                fieldnames += ['L_val']
        
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        # Set and compute some initial values
        index = 0
        if X_val is not None:
            index_val = 0

        for ite in range(iterations):
            # Get training and validation batches
            if (index + 1) * batch_size > X_train.shape[0]:
                X_batch = X_train[index * batch_size::]
                index = 0
            else:
                X_batch = X_train[index * batch_size:(index + 1) * batch_size]
                index += 1
            if X_val is not None:
                if (index_val + 1) * batch_size > X_val.shape[0]:
                    X_val_batch = X_val[index_val * batch_size::]
                    index_val = 0
                else:
                    X_val_batch = X_val[index_val * batch_size:(index_val + 1) * batch_size]
                    index_val += 1

            if self.model_type == 'desom':
                # Compute cluster assignments for batches
                _, d = self.model.predict(X_batch)
                y_pred = d.argmin(axis=1)
                if X_val is not None:
                    _, d_val = self.model.predict(X_val_batch)
                    y_val_pred = d_val.argmin(axis=1)

                # Update temperature parameter
                if ite < som_iterations:
                    if decay == 'exponential':
                        T = Tmax*(Tmin/Tmax)**(ite/(som_iterations-1))
                    elif decay == 'linear':
                        T = Tmax - (Tmax-Tmin)*(ite/(som_iterations-1))
            
                # Compute topographic weights batches
                w_batch = self.neighborhood_function(self.map_dist(y_pred), T)
                if X_val is not None:
                    w_val_batch = self.neighborhood_function(self.map_dist(y_val_pred), T)


                loss = self.model.train_on_batch(X_batch, [X_batch, w_batch])
                if self.AE_frozen:
                    loss = loss[2]
            
            elif self.model_type == 'ae':
                # Train on batch
                loss = self.model.train_on_batch(X_batch, [X_batch])

            if ite % eval_interval == 0:
                if self.model_type == 'desom':
                    # Initialize log dictionary
                    logdict = dict(iter=ite, T=T)

                    # Get SOM weights and decode to original space
                    decoded_prototypes = self.decode(self.prototypes)

                    # Evaluate losses and metrics
                    print('iteration {} - T={}'.format(ite, T))
                    if self.AE_frozen:
                        logdict['Lsom'] = loss
                        logdict['Lkm'] = kmeans_loss(y_pred, d)
                        logdict['Ltop'] = loss - logdict['Lkm']
                        logdict['latent_quantization_err'] = quantization_error(d)
                        logdict['latent_topographic_err'] = topographic_error(d, self.map_size)
                        d_original = np.square((np.expand_dims(X_batch, axis=1).reshape(-1, 1, self.encoder_dims[0]) - decoded_prototypes.reshape(-1,self.encoder_dims[0]))).sum(axis=2)
                        logdict['quantization_err'] = quantization_error(d_original) 
                        logdict['topographic_err'] = topographic_error(d_original, self.map_size)
                        print('[Train] - Lsom={:f} (Lkm={:f}/Ltop={:f})'.format(logdict['Lsom'], logdict['Lkm'], logdict['Ltop']))
                        print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err'], logdict['topographic_err']))
                        if X_val is not None:
                            val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch, w_val_batch])
                            val_loss = val_loss[2]
                            logdict['Lsom_val'] = val_loss
                            logdict['Lkm_val'] = kmeans_loss(y_val_pred, d_val)
                            logdict['Ltop_val'] = val_loss - logdict['Lkm_val']
                            logdict['latent_quantization_err_val'] = quantization_error(d_val)
                            logdict['latent_topographic_err_val'] = topographic_error(d_val, self.map_size)
                            d_original_val = np.square((np.expand_dims(X_batch, axis=1) - decoded_prototypes)).sum(axis=2)
                            logdict['quantization_err_val'] = quantization_error(d_original_val)
                            logdict['topographic_err_val'] = topographic_error(d_original_val, self.map_size)   
                            print('[Val] - Lsom={:f} (Lkm={:f}/Ltop={:f})'.format(logdict['Lsom_val'], logdict['Lkm_val'], logdict['Ltop_val']))
                            print('[Val] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err_val'], logdict['topographic_err_val']))

                    else:
                        logdict['L'] = loss[0]
                        logdict['Lr'] = loss[1]
                        logdict['Lsom'] = loss[2]
                        logdict['Lkm'] = kmeans_loss(y_pred, d)
                        logdict['Ltop'] = loss[2] - logdict['Lkm']
                        logdict['latent_quantization_err'] = quantization_error(d)
                        logdict['latent_topographic_err'] = topographic_error(d, self.map_size)
                        d_original = np.square((np.expand_dims(X_batch, axis=1).reshape(-1, 1, 1024) - decoded_prototypes.reshape(-1,1024))).sum(axis=2)
                        logdict['quantization_err'] = quantization_error(d_original) 
                        logdict['topographic_err'] = topographic_error(d_original, self.map_size)
                        print('[Train] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr'], logdict['Lsom'], logdict['Lkm'], logdict['Ltop'], logdict['L']))
                        print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err'], logdict['topographic_err']))
                        if X_val is not None:
                            val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch, w_val_batch])
                            logdict['L_val'] = val_loss[0]
                            logdict['Lr_val'] = val_loss[1]
                            logdict['Lsom_val'] = val_loss[2]
                            logdict['Lkm_val'] = kmeans_loss(y_val_pred, d_val)
                            logdict['Ltop_val'] = val_loss[2] - logdict['Lkm_val']
                            logdict['latent_quantization_err_val'] = quantization_error(d_val)
                            logdict['latent_topographic_err_val'] = topographic_error(d_val, self.map_size)
                            d_original_val = np.square((np.expand_dims(X_batch, axis=1) - decoded_prototypes)).sum(axis=2)
                            logdict['quantization_err_val'] = quantization_error(d_original_val)
                            logdict['topographic_err_val'] = topographic_error(d_original_val, self.map_size)   
                            print('[Val] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr_val'], logdict['Lsom_val'], logdict['Lkm_val'], logdict['Ltop_val'], logdict['L_val']))
                            print('[Val] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err_val'], logdict['topographic_err_val']))

                    logwriter.writerow(logdict)
                elif self.model_type == 'ae':
                    # Initialize log dictionary
                    logdict = dict(iter=ite)

                    # Evaluate losses and metrics
                    print('iteration {}'.format(ite))
                    logdict['L'] = loss
                    print('[Train] - total loss={:f}'.format(logdict['L']))
                    if X_val is not None:
                        val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch])
                        logdict['L_val'] = val_loss
                        print('[Val] - total loss={:f}'.format(logdict['L_val']))

                    logwriter.writerow(logdict)


            # Save intermediate model
            if ite % save_interval == 0:
                 self.model.save_weights(save_dir + '/{}_model_'.format(self.model_type) + str(ite) + '.h5')
                 print('Saved model to:', save_dir + '/{}_model_'.format(self.model_type) + str(ite) + '.h5')

        # Save the final model
        logfile.close()
        print('saving model to:', save_dir + '/{}_model_final.h5'.format(self.model_type))
        self.model.save_weights(save_dir + '/{}_model_final.h5'.format(self.model_type))


if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='data.csv', type=str, help='path of training set')
    parser.add_argument('--validation_size', default=0.25, type=float, help='represents the fraction in test split for validation [0,1)')
    parser.add_argument('--ae_weights', default=None, help='path of trained AE')
    parser.add_argument('--desom_weights', default=None, help='path of trained DESOM')
    parser.add_argument('--ae_freeze', default=0, type=int, help='freeze AE or not for SOM training after pre-train')
    parser.add_argument('--map_size', nargs='+', default=[25, 25], type=int, help='size of SOM map')
    parser.add_argument('--gamma', default=0.0075, type=float, help='coefficient of self-organizing map loss')
    parser.add_argument('--model_type', default='desom', choices=['desom', 'ae'], type=str, help="which model is building")
    parser.add_argument('--iterations', default=10000, type=int, help='number of iterations after pre-train')
    parser.add_argument('--som_iterations', default=10000, type=int, help='number of iterations for SOM completely decay')
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=257, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='model optimizer')
    parser.add_argument('--Tmax', default=10.0, type=float, help='initial decay constant in neighbour function')
    parser.add_argument('--Tmin', default=0.5, type=float, help='final decay constant in neighbour function')
    parser.add_argument('--decay', default='exponential', choices=['exponential', 'linear'], help='neighbour function type')
    parser.add_argument('--n_neurons', nargs='+', default=[1024, 32, 64, 128, 512, 200], type=int, help='list of neurons number')
    parser.add_argument('--n_dense', default=2, type=int, help='number of dense layer in the encoder')
    parser.add_argument('--decoder_type', default='transpose', choices=['transpose', 'upsampling'], type=str, help='layer type of decoder')
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    if args.ae_freeze == 1:
        args.ae_freeze = True
    else:
        args.ae_freeze = False
    print(args)

    stamp_sidelength = int(np.sqrt(args.n_neurons[0])) # side length of each input image

    # Load data
    (X_train, _), (X_val, _) = load_data(args.dataset, validation_size=args.validation_size)
    X_train = X_train.reshape(-1, stamp_sidelength, stamp_sidelength, 1)
    if ('X_val' in locals()):
        try:
            X_val = X_val.reshape(-1, stamp_sidelength, stamp_sidelength, 1)
        except AttributeError:
            pass

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

     # Instantiate model
    desom = DESOM(encoder_dims=args.n_neurons, num_dense=args.n_dense, map_size=args.map_size)
    
    # Initialize model
    desom.initialize(model_type=args.model_type, ae_act='relu', ae_init='glorot_uniform', ae_freeze=args.ae_freeze , decoder_type=args.decoder_type)
    plot_model(desom.model, to_file=os.path.join(args.save_dir, '{}_model.png'.format(args.model_type)), show_shapes=True)

    # Create log file
    with open('{}_log.txt'.format(args.model_type),'w') as f:
        for key in vars(args).keys():
            f.write('{}: {}\n'.format(key, vars(args)[key]))
    shutil.move('{}_log.txt'.format(args.model_type), args.save_dir)


    desom.model.summary()
    desom.compile(optimizer=args.optimizer, gamma=args.gamma)

    if (args.ae_weights is not None):
        desom.load_ae_weights(args.ae_weights)

    if (args.desom_weights is not None):
        desom.load_weights(args.desom_weights)

    # Fit model
    t0 = time()
    desom.fit(X_train, X_val, args.iterations, args.som_iterations, args.eval_interval,
              args.save_epochs, args.batch_size, args.Tmax, args.Tmin, args.decay, args.save_dir)
    print('Training time: ', (time() - t0))

    if args.model_type == 'desom':
        # Generate DESOM map visualization using reconstructed prototypes
        img_size = int(np.sqrt(args.n_neurons[0]))
        decoded_prototypes = desom.decode(desom.prototypes)
        fig, ax = plt.subplots(args.map_size[0], args.map_size[1], figsize=(10, 10))
        for k in range(args.map_size[0]*args.map_size[1]):
            ax[k // args.map_size[1]][k % args.map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size), vmin=0, vmax=1, cmap='gray')
            ax[k // args.map_size[1]][k % args.map_size[1]].axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(os.path.join(args.save_dir, 'desom_map.png'), bbox_inches='tight')
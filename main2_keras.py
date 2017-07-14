from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from model import *

import h5py
import itertools
import numpy as np
import csv
import os.path
import sys

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, MaxPooling2D, Flatten, Dropout
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.objectives import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

species_dir = str(sys.argv[1])
print(species_dir)
train_h5file = 'data/h5datasets_GO/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/h5datasets_GO/' + str(species_dir) + '/validation.h5'
promoter_length = 500

# training parameters
epochs = 15
batch_size = 200

# CNN hyperparameters
num_filters = 50
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.0001

# read in HDF5 file & create batch iterator for training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,batch_size)
train_batcher = train_data.batcher()

# read in HDF5 file for validation data
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(10000,validation_file.values()[0].shape[0])
validation_dat = validation_file['dnaseq'][0:val_datsize]
validation_labels = validation_file['labels'][0:val_datsize]

train_size = train_file.values()[0].shape[0]
num_GOterms = train_file['labels'].shape[1]

model = Sequential()
model.add(Conv2D(num_filters,[filter_height1,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_1', \
            input_shape=(4,promoter_length,1)))
model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
            name='AvgPool_1'))
model.add(Dropout(0.5))
model.add(Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_2'))
model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid', \
            name='AvgPool_2'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50,activation='relu',name='representation'))
model.add(Dense(num_GOterms,activation='sigmoid'))

# define one match accuracy measure
def one_match(y_true,y_pred):
    # one match accuracy
    onematch_pred = tf.equal(tf.argmax(tf.multiply(y_true,y_pred),axis=-1), \
        tf.argmax(y_pred,axis=-1))
    onematch = tf.reduce_mean(tf.cast(onematch_pred, tf.float32))
    return(onematch)

# compile model
model.compile(optimizer=Adam(lr=learning_rate),loss='binary_crossentropy', \
    metrics=['accuracy',one_match])

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

# set session and initialize all variables
sess = tf.Session()
K.set_session(sess)

print('epochs\tloss\tbinacc\tonematch\tval_binacc\tval_onematch')

for i in range(totalIterations):
    batch = train_batcher.next()

    model.train_on_batch(batch['dnaseq'],batch['labels'])

    # log training and validation accuracy
    if i%500 == 0:

        epoch_num = i/train_size*batch_size


        loss_val,binacc_val, onematch_val = model.test_on_batch(batch['dnaseq'],batch['labels'])

        val_loss_val,val_binacc_val, val_onematch_val = \
            model.test_on_batch(validation_dat,validation_labels)

        print('\t'.join([str(epoch_num),str(loss_val),str(binacc_val), \
            str(onematch_val),str(val_binacc_val),str(val_onematch_val)]))

# serialize model to JSON
model_json = model.to_json()
with open(species_dir + "_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(species_dir + "_model.h5")

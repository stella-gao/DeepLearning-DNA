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
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
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

# # create CallBack Tensorboard object
# tbCallBack = callbacks.TensorBoard(log_dir='./blah', histogram_freq=0, \
#         write_graph=True, write_images=True)

train_size = train_file.values()[0].shape[0]
num_GOterms = train_file['labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1))

# define placeholder for species labels
labels = tf.placeholder(tf.float32,shape=(None,num_GOterms))

# build layers of network
conv1 = Conv2D(num_filters,[filter_height1,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_1')(dna)
pool1 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
            name='AvgPool_1')(conv1)
drop1 = Dropout(0.5)(pool1)
conv2 = Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_2')(drop1)
pool2 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid', \
            name='AvgPool_2')(conv2)
drop2 = Dropout(0.5)(pool2)
flat = Flatten()(drop2)
FC = Dense(50,activation='relu',name='representation')(flat)
preds = Dense(num_GOterms,activation='softmax')(FC)

# loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# gradient descent optimizer (Adam)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(loss)

# one match accuracy
onematch_pred = tf.equal(tf.argmax(tf.multiply(labels,preds),axis=-1), \
        tf.argmax(preds,axis=-1))
onematch = tf.reduce_mean(tf.cast(onematch_pred, tf.float32))

# binary accuracy
binacc_pred = binary_accuracy(labels,preds)
binacc = K.mean(binacc_pred)

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

# initialize model saver
saver = tf.train.Saver(max_to_keep=4)

# set session and initialize all variables
sess = tf.Session()
K.set_session(sess)
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():

    print('epochs\tloss\tbinacc\tonematch\tval_binacc\tval_onematch')

    for i in range(totalIterations):
        batch = train_batcher.next()
        sess.run([train_step],feed_dict={dna: batch['dnaseq'], \
            labels: batch['labels'], K.learning_phase(): 1})

        # log training and validation accuracy
        if i%500 == 0:

            epoch_num = i/train_size*batch_size

            loss_val,onematch_val = sess.run([loss,onematch],\
                feed_dict={dna: batch['dnaseq'], \
                labels: batch['labels'], K.learning_phase(): 0})

            binacc_val = binacc.eval(feed_dict={dna: batch['dnaseq'], \
                labels: batch['labels'], K.learning_phase(): 0})
            val_binacc_val = binacc.eval(feed_dict={dna: validation_dat, \
                labels: validation_labels, K.learning_phase(): 0})

            val_onematch_val = sess.run(onematch, \
                feed_dict={dna: validation_dat, \
                labels: validation_labels,K.learning_phase(): 0})

            print('\t'.join([str(epoch_num),str(loss_val),str(binacc_val), \
                str(onematch_val),str(val_binacc_val),str(val_onematch_val)]))

    # save model
    saver.save(sess,species_dir + '_model')

    # get representational output 
    rep_layer = sess.run(FC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(species_dir + '_rep.txt',rep_layer,delimiter='\t')
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
from keras.layers.advanced_activations import LeakyReLU
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

def conv_drop(input_tensor,layer_num,filter_height):
    conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', \
                kernel_regularizer='l2',padding='valid',name='conv_'+str(layer_num))(input_tensor)
    leak = LeakyReLU(alpha=.001)(conv)
    drop = Dropout(0.5)(leak)

    return drop

def conv_pool_drop(input_tensor,layer_num,filter_height):
    conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', \
                kernel_regularizer='l2',padding='valid',name='conv_'+str(layer_num))(input_tensor)
    leak = LeakyReLU(alpha=.001)(conv)
    pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
                name='AvgPool_'+str(layer_num))(leak)
    drop = Dropout(0.5)(pool)

    return drop

species_dir = str(sys.argv[1])
print(species_dir)

train_h5file = 'data/simulated_dna/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/simulated_dna/' + str(species_dir) + '/validation.h5'

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

promoter_length = validation_dat.shape[2]

# create CallBack Tensorboard object
tbCallBack = callbacks.TensorBoard(log_dir='./blah', histogram_freq=0, \
        write_graph=True, write_images=True)

train_size = train_file.values()[0].shape[0]
num_species = train_file['labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='dna')

# define placeholder for species labels
labels = tf.placeholder(tf.float32,shape=(None,num_species),name='label')

# build layers of network
drop1 = conv_drop(dna,1,filter_height1)
drop2 = conv_drop(drop1,2,filter_height2)
drop3 = conv_pool_drop(drop2,3,filter_height2)
# drop4 = conv_drop(drop3,4,filter_height2)
# drop5 = conv_drop(drop4,5,filter_height2)
# drop6 = conv_pool_drop(drop5,6,filter_height2)

flat = Flatten()(drop3)
FC = Dense(50,activation='relu',name='representation')(flat)
preds = Dense(num_species,activation='softmax')(FC)

# loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# gradient descent optimizer (Adam)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(loss)

# internal accuracy
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

# set session and initialize all variables
sess = tf.Session()
K.set_session(sess)
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=2)

    lowest_val_loss = 1000

    print('epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss')
    for i in range(totalIterations):
        batch = train_batcher.next()
        sess.run([train_step],feed_dict={dna: batch['dnaseq'], \
            labels: batch['labels'], K.learning_phase(): 1})

        # log training and validation accuracy
        if i%1000 == 0:

            epoch_num = i/train_size*batch_size

            train_acc,train_loss = sess.run([accuracy,loss],feed_dict={dna: batch['dnaseq'], \
                labels: batch['labels'], K.learning_phase(): 0})

            val_acc,val_loss = sess.run([accuracy,loss],feed_dict={dna: validation_dat, \
                labels: validation_labels, K.learning_phase(): 0})

            print('\t'.join([str(epoch_num),str(train_acc),str(train_loss), \
                str(val_acc),str(val_loss)]))

            # save model if current validation loss is lower than the previous lowest
            if val_loss < lowest_val_loss:
                saver.save(sess,species_dir + '_simDNAmodel')
                lowest_val_loss = val_loss

    # get representational output 
    rep_layer = sess.run(FC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(species_dir + '_simDNArep.txt',rep_layer,delimiter='\t')


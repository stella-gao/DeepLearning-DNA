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
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

species_dir = str(sys.argv[1])
print(species_dir)
train_h5file = 'data/h5datasets_GO/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/h5datasets_GO/' + str(species_dir) + '/validation.h5'
promoter_length = 500

# training parameters
epochs = 20
batch_size = 20

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
validation_labels = validation_file['GO_labels'][0:val_datsize]

# # create CallBack Tensorboard object
# tbCallBack = callbacks.TensorBoard(log_dir='./blah', histogram_freq=0, \
#         write_graph=True, write_images=True)

train_size = train_file.values()[0].shape[0]
num_GOterms = 1 #train_file['GO_labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='dna')

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

conv3 = Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_2')(drop2)
pool3 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid', \
            name='AvgPool_2')(conv3)
drop3 = Dropout(0.5)(pool3)

# conv4 = Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_2')(drop3)
# pool4 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid', \
#             name='AvgPool_2')(conv4)
# drop4 = Dropout(0.5)(pool4)

# conv5 = Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_2')(drop4)
# pool5 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid', \
#             name='AvgPool_2')(conv5)
# drop5 = Dropout(0.5)(pool5)

flat = Flatten()(drop3)
FC = Dense(50,activation='relu',name='representation')(flat)

preds_list = [Dense(2,activation='softmax')(FC) for i in range(num_GOterms)]

# define placeholders for 
labels_list = [tf.placeholder(tf.float32,[None,2]) for i in range(num_GOterms)]

total_loss = 0
for i in range(num_GOterms):
    total_loss += tf.reduce_mean(categorical_crossentropy(labels_list[i],preds_list[i]))

# gradient descent optimizer (Adam)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(total_loss)

# # one match accuracy
# onematch_pred = tf.equal(tf.argmax(tf.multiply(labels,preds),axis=-1), \
#         tf.argmax(preds,axis=-1))
# onematch = tf.reduce_mean(tf.cast(onematch_pred, tf.float32))

# exact match accuracy
# match = tf.equal(float(num_GOterms),\
#     tf.reduce_sum(tf.cast(tf.equal(labels,tf.round(preds)),tf.float32),axis=1))
# exactmatch = tf.reduce_mean(tf.cast(match,tf.float32))
match = 0
for i in range(num_GOterms):
    match += tf.cast(tf.equal(tf.argmax(preds_list[i], 1), tf.argmax(labels_list[i], 1)),tf.float32)
exactmatch = tf.reduce_mean(tf.cast(tf.equal(float(num_GOterms),match),tf.float32))

# binary accuracy
acc = 0
for i in range(num_GOterms):
    correct_pred = tf.equal(tf.argmax(preds_list[i], 1), tf.argmax(labels_list[i], 1))
    acc += tf.divide(tf.reduce_mean(tf.cast(correct_pred, tf.float32)),num_GOterms)
    # acc += tf.divide(categorical_accuracy(labels_list[i],preds_list[i]),num_GOterms)
# binacc = tf.reduce_mean(binacc_vals)

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

# set session and initialize all variables
sess = tf.Session()
K.set_session(sess)
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=4)

    print('epochs\ttrain_loss\ttrain_acc\ttrain_exactmatch\tval_loss\tval_acc\tval_exactmatch')

    for i in range(totalIterations):
        batch = train_batcher.next()

        train_feed_dict = {labels_list[i]: batch['GO_labels'][:,i] for i in range(num_GOterms)}
        train_feed_dict.update({dna: batch['dnaseq'], K.learning_phase(): 1})

        sess.run([train_step],feed_dict=train_feed_dict)

        # log training and validation accuracy
        if i%100 == 0:

            epoch_num = i/train_size*batch_size

            train_feed_dict = {labels_list[i]: batch['GO_labels'][:,i] for i in range(num_GOterms)}
            train_feed_dict.update({dna: batch['dnaseq'], K.learning_phase(): 0})
            train_loss,train_exactmatch,train_acc = sess.run([total_loss,exactmatch,acc],\
                feed_dict=train_feed_dict)

            val_feed_dict = {labels_list[i]: validation_labels[:,i] for i in range(num_GOterms)}
            val_feed_dict.update({dna: validation_dat, K.learning_phase(): 0})

            val_loss,val_exactmatch,val_acc = sess.run([total_loss,exactmatch,acc], \
                feed_dict=val_feed_dict)

            print('\t'.join([str(epoch_num),str(train_loss),str(train_acc), \
                str(train_exactmatch),str(val_loss),str(val_acc),str(val_exactmatch)]))

    # save model
    saver.save(sess,species_dir + '_model')

    # get representational output 
    val_feed_dict = {labels_list[i]: validation_labels[:,i] for i in range(num_GOterms)}
    val_feed_dict.update({dna: validation_dat, K.learning_phase(): 0})

    rep_layer = sess.run(FC,feed_dict=val_feed_dict)

    # write representational output to file
    np.savetxt(species_dir + '_rep.txt',rep_layer,delimiter='\t')
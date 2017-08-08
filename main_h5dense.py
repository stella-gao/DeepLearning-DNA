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
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

def dense_block(input_tensor,layer_num,num_filters,filter_width,kernelReg=None):

    conv1 = Conv2D(num_filters,[1,filter_width],activation='linear', \
                kernel_regularizer=kernelReg,padding='same',name='denseconv1_' + str(layer_num))(input_tensor)
    leak1 = LeakyReLU(alpha=.001)(conv1)

    drop1 = Dropout(0.5)(leak1)

    # concatenate outputs
    concat1 = tf.concat([input_tensor,drop1],3)

    conv2 = Conv2D(num_filters,[2,filter_width],activation='linear', \
                kernel_regularizer=kernelReg,padding='same',name='denseconv2_' + str(layer_num))(concat1)
    leak2 = LeakyReLU(alpha=.001)(conv2)
    drop2 = Dropout(0.5)(leak2)

    # concatenate outputs
    concat2 = tf.concat([concat1,drop2],3)

    conv3 = Conv2D(num_filters,[3,filter_width],activation='linear', \
                kernel_regularizer=kernelReg,padding='same',name='denseconv3_' + str(layer_num))(concat2)
    leak3 = LeakyReLU(alpha=.001)(conv3)
    drop3 = Dropout(0.5)(leak3)

    return drop3

species_dir = str(sys.argv[1])
print(species_dir)
train_h5file = 'data/h5datasets/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/h5datasets/' + str(species_dir) + '/validation.h5'
promoter_length = 500

# training parameters
epochs = 30
batch_size = 200

# CNN hyperparameters
num_filters = 80
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 2
learning_rate = 0.0001

# print hyperparameters
print('Batch Size: ' + str(batch_size))
print('Number of Filters: ' + str(num_filters))
print('Filter Width: ' + str(filter_width))

# read in HDF5 file & create batch iterator for training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,batch_size)
train_batcher = train_data.batcher()

# read in HDF5 file for validation data
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(1000,validation_file.values()[0].shape[0])
validation_dat = validation_file['dnaseq'][0:val_datsize]
validation_labels = validation_file['species_labels'][0:val_datsize]

# create CallBack Tensorboard object
tbCallBack = callbacks.TensorBoard(log_dir='./blah', histogram_freq=0, \
        write_graph=True, write_images=True)

train_size = train_file.values()[0].shape[0]
num_species = train_file['species_labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='dna')

# define placeholder for species labels
labels = tf.placeholder(tf.float32,shape=(None,num_species),name='label')

# build layers of network

# transition layer
trans1_conv = Conv2D(num_filters,[filter_height1,filter_width],activation='linear', \
            kernel_regularizer=None,padding='valid',name='trans1_conv')(dna)
trans1_leak = LeakyReLU(alpha=.001)(trans1_conv)
trans1_pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
                name='AvgPool_trans1')(trans1_leak)
trans1_drop = Dropout(0.5)(trans1_pool)

# dense block
denseblock1 = dense_block(trans1_drop,1,num_filters=50,filter_width=filter_width)

# transition layer
trans2_conv = Conv2D(num_filters,[filter_height2,filter_width],activation='linear', \
            kernel_regularizer=None,padding='valid',name='trans2_conv')(denseblock1)
trans2_leak = LeakyReLU(alpha=.001)(trans2_conv)
trans2_pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
                name='AvgPool_trans2')(trans2_leak)
trans2_drop = Dropout(0.5)(trans2_pool)               


flat = Flatten()(trans2_drop)
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

f = open(species_dir + '_dense.output.txt','w')
writer = csv.writer(f,delimiter='\t')
with sess.as_default():

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=2)

    lowest_val_loss = 1000

    header = 'epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss'
    writer.writerow(['epochs','train_acc','train_loss','val_acc','val_loss'])
    print(header)
    for i in range(totalIterations):
        batch = train_batcher.next()
        sess.run([train_step],feed_dict={dna: batch['dnaseq'], \
            labels: batch['species_labels'], K.learning_phase(): 1})

        # log training and validation accuracy
        if i%1000 == 0:

            epoch_num = i/train_size*batch_size

            train_acc,train_loss = sess.run([accuracy,loss],feed_dict={dna: batch['dnaseq'], \
                labels: batch['species_labels'], K.learning_phase(): 0})

            val_acc,val_loss = sess.run([accuracy,loss],feed_dict={dna: validation_dat, \
                labels: validation_labels, K.learning_phase(): 0})

            print('\t'.join([str(epoch_num),str(train_acc),str(train_loss), \
                str(val_acc),str(val_loss)]))
            writer.writerow([epoch_num,train_acc,train_loss,val_acc,val_loss])

            # save model if current validation loss is lower than the previous lowest
            if val_loss < lowest_val_loss:
                saver.save(sess,species_dir + '_dense_model')
                lowest_val_loss = val_loss

    f.close()
    # get representational output 
    rep_layer = sess.run(FC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(species_dir + '_dense_rep.txt',rep_layer,delimiter='\t')


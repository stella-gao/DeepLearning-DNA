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
from keras.layers import Input, Dense, Lambda, Conv1D, Conv2D, concatenate, Reshape, AveragePooling2D, MaxPooling2D, Flatten, Dropout, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

promoter_length = 500

# training parameters
epochs = 5
batch_size = 50

# CNN hyperparameters
num_filters = 80
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.0001

species_dir = str(sys.argv[1])
outfile_name = species_dir + ''
print(species_dir)

train_h5file = 'data/h5datasets/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/h5datasets/' + str(species_dir) + '/validation.h5'

# print hyperparameters
print('Batch Size: ' + str(batch_size))
print('Number of Filters: ' + str(num_filters))
print('Filter Width: ' + str(filter_width))

def conv_drop(input_tensor,layer_num,filter_height,filter_width,kernelReg=None,dilation=False):
    if dilation:
        conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', \
                kernel_regularizer=kernelReg,padding='valid',dilation_rate=dilation,\
                name='dilconv_'+str(layer_num))(input_tensor)
    else:
        conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', 
                kernel_regularizer=kernelReg,padding='valid',name='conv_'+str(layer_num))(input_tensor)
    leak = LeakyReLU(alpha=.001)(conv)
    drop = Dropout(0.5)(leak)

    return drop

def conv_pool_drop(input_tensor,layer_num,filter_height,filter_width,kernelReg=None,dilation=False):
    if dilation:
        conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', \
                kernel_regularizer=kernelReg,padding='valid',dilation_rate=dilation,\
                name='dilconv_'+str(layer_num))(input_tensor)
        leak = LeakyReLU(alpha=.001)(conv)
        pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
                    name='dilAvgPool_'+str(layer_num))(leak) 
    else:
        conv = Conv2D(num_filters,[filter_height,filter_width],activation='linear', 
                kernel_regularizer=kernelReg,padding='valid',name='conv_'+str(layer_num))(input_tensor)
        leak = LeakyReLU(alpha=.001)(conv)
        pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
                    name='AvgPool_'+str(layer_num))(leak)
    drop = Dropout(0.5)(pool)

    return drop

def LSTMConvBranches(input_tensor,filter_width,num_units,kernelReg=None,dilation=None):

    # LSTM branch
    lstm1 = LSTM(num_units,implementation=2)(tf.squeeze(tf.reshape(dna,\
        [tf.shape(dna)[0],tf.shape(dna)[2],4,1]),[3]))
    lstm_drop1 = Dropout(0.5)(lstm1)

    # Convolutional branch
    drop1 = conv_pool_drop(input_tensor,1,4,filter_width,kernelReg=kernelReg,dilation=dilation)
    drop2 = conv_pool_drop(drop1,2,1,filter_width,kernelReg=kernelReg,dilation=dilation)
    flat1 = Flatten()(drop2)
    convFC = Dense(num_units,activation='linear')(flat1)

    # stack LSTM and convolutional layer outputs
    stacked_layers = tf.stack([lstm_drop1,convFC],1)
    stacked_layers = tf.expand_dims(stacked_layers,axis=3)
    print(stacked_layers)

    # convolutional layer
    conv_stacked = conv_pool_drop(stacked_layers,'stack',2,filter_width)

    return conv_stacked

# read in HDF5 file & create batch iterator for training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,batch_size)
train_batcher = train_data.batcher()

# read in HDF5 file for validation data
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(500,validation_file.values()[0].shape[0])
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

###############################################################################
### MODEL #####################################################################
###############################################################################

# # LSTM layer
# lstm1 = LSTM(128,implementation=2)(tf.squeeze(drop1,[1]))
# lstm_drop1 = Dropout(0.5)(lstm1)

# # reshape input tensor for LSTM
lstm1 = LSTM(128,return_sequences=True,implementation=2)(tf.squeeze(tf.reshape(dna,[tf.shape(dna)[0],promoter_length,4,1]),[3]))
lstm_drop1 = Dropout(0.5)(lstm1)
# lstm2 = LSTM(128,return_sequences=True,implementation=2)(lstm_drop1)
# lstm_drop2 = Dropout(0.5)(lstm2)

# drop1 = conv_pool_drop(dna,1,filter_height1,filter_width)
# drop2 = conv_pool_drop(drop1,2,filter_height2,filter_width)
# drop3 = conv_pool_drop(drop2,3,filter_height2,filter_width)

# lstm1 = LSTM(128,return_sequences=True,implementation=2)(tf.squeeze(drop1,[1]))
# lstm_drop1 = Dropout(0.5)(lstm1)

# lstm_conv = LSTMConvBranches(dna,filter_width,128)

flat = Flatten()(lstm_drop1)
FC = Dense(50,activation='relu',name='representation')(flat)
preds = Dense(num_species,activation='softmax')(FC)

###############################################################################
###############################################################################
###############################################################################

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

    # initialize writer to write output to file
    f = open(outfile_name + '.output','w')
    writer = csv.writer(f,delimiter='\t')

    print('epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss')
    writer.writerow(['epochs','train_acc','train_loss','loss_acc','val_loss'])

    f.close()

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

            f = open(outfile_name + '.output','a')
            writer = csv.writer(f,delimiter='\t')
            writer.writerow([epoch_num,train_acc,train_loss,val_acc,val_loss])
            f.close()

            # save model if current validation loss is lower than the previous lowest
            if val_loss < lowest_val_loss:
                saver.save(sess,outfile_name + '_model')
                lowest_val_loss = val_loss

    # get representational output 
    rep_layer = sess.run(FC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(outfile_name + '_rep.txt',rep_layer,delimiter='\t')

f.close()

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
epochs = 30
batch_size = 200

# CNN hyperparameters
num_filters = 50
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.0001

# LSTM hyperparameters
n_steps = 50 # timesteps 
n_hidden = 128 # hidden layer num of features

species_dir = str(sys.argv[1])
outfile_name = species_dir + 'LSTMsmall'
print(species_dir)

train_h5file = 'data/h5datasets/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/h5datasets/' + str(species_dir) + '/validation.h5'

# print hyperparameters
print('Batch Size: ' + str(batch_size))
print('Number of Filters: ' + str(num_filters))
print('Filter Width: ' + str(filter_width))

def conv_drop(input_tensor,layer_num,filter_height,filter_width,num_filters,kernelReg=None,dilation=False):
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

def conv_pool_drop(input_tensor,layer_num,filter_height,filter_width,num_filters,kernelReg=None,dilation=False):
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

def RNN(x,n_steps,n_hidden,n_classes,celltype='LSTM',return_seq=False):
    # input shape: (batch_size, n_steps, n_input)
    # converted shape:'n_steps' tensors list of shape (batch_size, n_input)

    # define weights, biases for LSTM component
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    if celltype == 'LSTM':
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    elif celltype == 'GRU':
        cell = tf.contrib.rnn.GRUCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)

    if return_seq:
        return [tf.matmul(outp, weights['out']) + biases['out'] for outp in outputs]
    else:
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

def multiRNN(x,n_steps,n_hidden,n_classes,numLayers,return_seq=False):
    # input shape: (batch_size, n_steps, n_input)
    # converted shape:'n_steps' tensors list of shape (batch_size, n_input)

    # define weights, biases for LSTM component
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0) \
        for i in range(numLayers)])
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    if return_seq:
        return [tf.matmul(outp, weights['out']) + biases['out'] for outp in outputs]
    else:
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

def LSTMConvBranches(input_tensor,filter_width,num_filters,kernelReg=None,dilation=None):

    # LSTM branch
    conv1 = Conv2D(8,[4,5],activation='linear',
                    name='convTrans_1',padding='valid')(dna)
    leak1 = LeakyReLU(alpha=.001)(conv1)
    pool1 = AveragePooling2D((1,2),strides=(1,2),name='AvgPoolTrans_1',padding='same')(leak1) 
    conv2 = Conv2D(1,[1,1],activation='linear',name='convTrans1x1',padding='same')(pool1)
    reshapedSeq = tf.reshape(tf.squeeze(conv2,[1,3]),[tf.shape(conv2)[0],31,8])
    lstm1 = RNN(reshapedSeq,n_steps)

    # Convolutional branch
    drop1 = conv_pool_drop(input_tensor,1,4,filter_width,num_filters,kernelReg=kernelReg,dilation=dilation)
    drop2 = conv_pool_drop(drop1,2,1,filter_width,num_filters,kernelReg=kernelReg,dilation=dilation)
    flat1 = Flatten()(drop2)
    convFC = Dense(num_units,activation='linear')(flat1)

    # stack LSTM and convolutional layer outputs
    stacked_layers = tf.stack([lstm_drop1,convFC],1)
    stacked_layers = tf.expand_dims(stacked_layers,axis=3)
    print(stacked_layers)

    # convolutional layer
    conv_stacked = conv_pool_drop(stacked_layers,'stack',2,filter_width,num_filters)

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

# get training sample size and number of species
train_size = train_file.values()[0].shape[0]
num_species = train_file['species_labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='dna')

# define placeholder for species labels
labels = tf.placeholder(tf.float32,shape=(None,num_species),name='label')

###############################################################################
### MODEL #####################################################################
###############################################################################

# # LSTM
# reshapedDNA = tf.concat(tf.split(tf.squeeze(dna,[3]),int(promoter_length/n_steps),2),0)
# reshapedDNA = tf.reshape(reshapedDNA,[tf.shape(reshapedDNA)[0],tf.shape(reshapedDNA)[2],4])
# with tf.variable_scope('LSTM1'):
#     lstm1 = RNN(reshapedDNA,n_steps,n_hidden,100)

# n_steps2 = int(promoter_length/n_steps)
# reshapedLSTM1 = tf.reshape(lstm1,[tf.shape(dna)[0],n_steps2,100])
# with tf.variable_scope('LSTM2'):
#     lstm2 = RNN(reshapedLSTM1,n_steps2,n_hidden,100)

reshapedDNA = tf.reshape(tf.squeeze(dna,[3]),[tf.shape(dna)[0],n_steps,int(promoter_length/n_steps)*4])
print(reshapedDNA)
with tf.variable_scope('RNN1'):
    rnn1 = RNN(reshapedDNA,n_steps,n_hidden,32,celltype='GRU')
    print(rnn1)

# lstm1 = multiRNN(reshapedDNA,n_steps,n_hidden,32,3)

# n_steps2 = int(promoter_length/n_steps)
# reshapedLSTM1 = tf.reshape(lstm1,[tf.shape(dna)[0],n_steps2,32])

# reshapedLSTM1 = tf.reshape(reshapedLSTM1,[tf.shape(reshapedLSTM1)[0],32,n_steps2])
# drop1 = conv_pool_drop(tf.expand_dims(reshapedLSTM1,3),2,4,filter_width,num_filters)
# drop2 = conv_pool_drop(drop1,2,4,filter_width,num_filters)

# LSTMFlat = Flatten()(lstm2)
# LSTMRep = Dense(100,activation='relu',name='LSTMRep')(lstm2)

# lstm1 = RNN(tf.squeeze(tf.reshape(dna,[tf.shape(dna)[0],tf.shape(dna)[2],4,1]),[3]),\
#     weights,biases)
# lstm_drop1 = Dropout(0.5)(lstm1)

# LSTM branch
# conv1 = Conv2D(64,[4,5],activation='linear',
#                 name='convTrans_1',padding='valid')(dna)
# leak1 = LeakyReLU(alpha=.001)(conv1)
# pool1 = AveragePooling2D((1,2),strides=(1,2),name='AvgPoolTrans_1',padding='same')(leak1) 
# conv2 = Conv2D(1,[1,1],activation='linear',name='convTrans1x1',padding='same')(pool1)
# print(conv2)
# reshapedSeq = tf.reshape(tf.squeeze(conv2,[1,3]),[tf.shape(conv2)[0],31,8])
# lstm1 = RNN(reshapedSeq,n_steps,n_hidden,100)

# lstm_stacked = multiRNN(reshapedSeq,n_steps,n_hidden,100,2)

# # # convolutional branch
# drop1 = conv_pool_drop(dna,1,filter_height1,filter_width,num_filters)
# drop2 = conv_pool_drop(drop1,2,filter_height2,filter_width,num_filters)
# # drop3 = conv_pool_drop(drop2,3,filter_height2,filter_width,num_filters)
# ConvFlat = Flatten()(drop2)
# ConvRep = Dense(100,activation='relu',name='ConvRep')(ConvFlat)

# stack LSTM and convolutional branch outputs
# stacked_layers = tf.stack([LSTMRep,ConvRep],1)
# stacked_layers = tf.stack([lstm1,ConvRep],1)

# flat = Flatten()(drop2)
FC = Dense(50,activation='relu',name='representation')(rnn1)
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

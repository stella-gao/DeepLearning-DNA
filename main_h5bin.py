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
epochs = 20
batch_size = 200

# CNN hyperparameters
num_filters = 50
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.0001

species_dir = 'all10bin' #str(sys.argv[1])
outfile_name = species_dir + 'Binary'
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

def predPrecisionRecall(pred_probs_list,binlabel_arr):

    preds = [1*(pred_arr[:,0] > 0.5) for pred_arr in pred_probs_list]
    truePos = [preds[i]*binlabel_arr[:,i,0] for i in range(len(preds))]
    recall = [sum(truePos[i])/sum(binlabel_arr[:,i,0]) for i in range(len(preds))]
    precision = [sum(truePos[i])/sum(preds[i]) for i in range(len(preds))]

    return np.nanmean(precision), np.nanmean(recall)

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

# get training sample size and number of species
train_size = train_file.values()[0].shape[0]
num_species = train_file['species_labels'].shape[1]

# define placeholder for DNA sequences (represented via one-hot encoding)
dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='dna')

###############################################################################
### MODEL #####################################################################
###############################################################################

# build layers of network
drop1 = conv_pool_drop(dna,1,filter_height1,filter_width,num_filters)
drop2 = conv_pool_drop(drop1,2,filter_height2,filter_width,num_filters)
drop3 = conv_pool_drop(drop2,3,filter_height2,filter_width,num_filters)
drop4 = conv_pool_drop(drop2,4,filter_height2,filter_width,num_filters)

flat = Flatten()(drop4)
allFC = Dense(100,activation='relu',name='representationAll')(flat)
FC_list = [Dense(50,activation='relu',name='representation_'+str(i))(allFC) \
    for i in range(num_species)]
preds_list = [Dense(2,activation='softmax')(FC_list[i]) for i in range(num_species)]

###############################################################################
###############################################################################
###############################################################################

# define placeholders for binary species labels
labels_list = [tf.placeholder(tf.float32,[None,2],name='label' + str(i)) for i in range(num_species)]

# total loss across all binary labels
total_loss = 0
for i in range(num_species):
    total_loss += tf.reduce_mean(categorical_crossentropy(labels_list[i],preds_list[i]))

# gradient descent optimizer (Adam)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(total_loss)

# % of examples w/ correct predictions across all species labels
match = 0
for i in range(num_species):
    match += tf.cast(tf.equal(tf.argmax(preds_list[i], 1), tf.argmax(labels_list[i], 1)),tf.float32)
exactmatch_acc = tf.reduce_mean(tf.cast(tf.equal(float(num_species),match),tf.float32))

# % of examples w/ correct predictions for the assigned label
onematch = 0
for i in range(num_species):
    predTrue = tf.cast(tf.equal(tf.argmax(preds_list[i],1),0),tf.float32)
    onematch += tf.reduce_sum(tf.multiply(predTrue,tf.gather(tf.transpose(labels_list[i]),[0])))
onematch_acc = tf.divide(onematch,tf.cast(tf.shape(preds_list[0])[0],tf.float32))

# onematch_pred = tf.equal(tf.argmax(tf.multiply(labels,preds),axis=-1), \
#         tf.argmax(preds,axis=-1))
# onematch = tf.reduce_mean(tf.cast(onematch_pred, tf.float32))

# # sum of categorical accuracy values
# acc = 0
# for i in range(num_species):
#     correct_pred = tf.equal(tf.argmax(preds_list[i], 1), tf.argmax(labels_list[i], 1))
#     acc += tf.divide(tf.reduce_mean(tf.cast(correct_pred, tf.float32)),num_species)

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

    # # initialize writer to write output to file
    # f = open(outfile_name + '.output','w')
    # writer = csv.writer(f,delimiter='\t')

    print('epochs\ttrain_exactmatch\ttrain_prec\ttrain_recall\ttrain_loss\tval_exactmatch\tval_prec\tval_recall\tval_loss')
    # writer.writerow(['epochs','train_exactmatch','train_prec','train_recall','train_loss','val_exactmatch','val_prec','val_recall','val_loss'])

    # f.close()

    for i in range(totalIterations):
        batch = train_batcher.next()
        train_feed_dict = {labels_list[j]: batch['species_labels'][:,j] for j in range(num_species)}
        train_feed_dict.update({dna: batch['dnaseq'], K.learning_phase(): 1})

        sess.run([train_step],feed_dict=train_feed_dict)

        # log training and validation accuracy
        if i%1000 == 0:

            epoch_num = i/train_size*batch_size

            # training set feed dictionary
            train_feed_dict = {labels_list[j]: batch['species_labels'][:,j] for j in range(num_species)}
            train_feed_dict.update({dna: batch['dnaseq'], K.learning_phase(): 0})

            # training set exact match accuracy, loss
            train_exactmatchacc,train_loss = sess.run([exactmatch_acc,\
                total_loss],feed_dict=train_feed_dict)

            # calculate training set precision, recall
            train_preds = sess.run(preds_list,feed_dict=train_feed_dict)
            train_prec, train_recall = predPrecisionRecall(train_preds,batch['species_labels'])

            # validation set feed dictionary
            val_feed_dict = {labels_list[i]: validation_labels[:,i] for i in range(num_species)}
            val_feed_dict.update({dna: validation_dat, K.learning_phase(): 0})

            # validation set exact match accuracy, loss
            val_exactmatchacc,val_loss = sess.run([exactmatch_acc,\
                total_loss],feed_dict=val_feed_dict)

            # calculate validation set precision, recall
            val_preds = sess.run(preds_list,feed_dict=val_feed_dict)
            val_prec, val_recall = predPrecisionRecall(val_preds,validation_labels)

            print('\t'.join([str(epoch_num),str(train_exactmatchacc),str(train_prec),\
                str(train_recall),str(train_loss),str(val_exactmatchacc),str(val_prec), \
                str(val_recall),str(val_loss)]))

            # f = open(outfile_name + '.output','a')
            # writer = csv.writer(f,delimiter='\t')
            # writer.writerow([epoch_num,train_exactmatchacc,train_prec,train_recall,train_loss,\
            #     val_exactmatchacc,val_prec,val_recall,val_loss])
            # f.close()

            # # save model if current validation loss is lower than the previous lowest
            # if val_loss < lowest_val_loss:
            #     saver.save(sess,outfile_name + '_model')
            #     lowest_val_loss = val_loss

    # get representational output 
    rep_layer = sess.run(allFC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(outfile_name + '_rep.txt',rep_layer,delimiter='\t')

f.close()

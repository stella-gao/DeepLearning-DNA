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

species_dir = 'all10bin'
model_dir = ''
model_name = 'all10binBinary_model'
train_h5file = 'data/h5datasets/all10bin/train.h5'
validation_h5file = 'data/h5datasets/all10bin/validation.h5'

outfile_name = 'all10bin'

# training parameters
epochs = 25
batch_size = 50

# read in HDF5 file & create batch iterator for training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,batch_size)
train_batcher = train_data.batcher()

# read in HDF5 file for validation data
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(1000,validation_file.values()[0].shape[0])
validation_dat = validation_file['dnaseq'][0:val_datsize]
validation_labels = validation_file['species_labels'][0:val_datsize]

# # create CallBack Tensorboard object
# tbCallBack = callbacks.TensorBoard(log_dir='./blah', histogram_freq=0, \
#         write_graph=True, write_images=True)

train_size = train_file.values()[0].shape[0]
num_species = train_file['species_labels'].shape[1]

# restore graph
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
saver.restore(sess,'./' + model_dir + model_name)

graph = tf.get_default_graph()

# identify relevant placeholders and operations
dna = graph.get_tensor_by_name("dna:0")
dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
rep = graph.get_tensor_by_name('representationAll/Relu:0')
preds_list = [graph.get_tensor_by_name('dense_'+str(i+1)+'/Softmax:0') for i in range(num_species)]
train_step = graph.get_operation_by_name('Adam')

# define placeholders for binary species labels
labels_list = [graph.get_tensor_by_name("label" + str(i) + ":0") for i in range(num_species)]
# labels_list = [graph.get_tensor_by_name("Placeholder:0")]
# labels_list.extend([graph.get_tensor_by_name("Placeholder_"+str(i)+":0") for i in range(1,num_species)])

# total loss across all binary labels
total_loss = 0
for i in range(num_species):
    total_loss += tf.reduce_mean(categorical_crossentropy(labels_list[i],preds_list[i]))

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

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

with sess.as_default():

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=2)

    lowest_val_loss = 1000

    # initialize writer to write output to file
    f = open(outfile_name + '.output','w')
    writer = csv.writer(f,delimiter='\t')

    print('epochs\ttrain_onematch\ttrain_exactmatch\ttrain_loss\tval_onematch\tval_exactmatch\tval_loss')
    writer.writerow(['epochs','train_onematch','train_exactmatch','train_loss','val_onematch','val_exactmatch','val_loss'])

    f.close()

    for i in range(totalIterations):
        batch = train_batcher.next()
        train_feed_dict = {labels_list[j]: batch['species_labels'][:,j] for j in range(num_species)}
        train_feed_dict.update({dna: batch['dnaseq'], dropout1: 1})

        sess.run([train_step],feed_dict=train_feed_dict)

        # log training and validation accuracy
        if i%1000 == 0:

            epoch_num = i/train_size*batch_size

            # training set feed dictionary
            train_feed_dict = {labels_list[j]: batch['species_labels'][:,j] for j in range(num_species)}
            train_feed_dict.update({dna: batch['dnaseq'], dropout1: 0})

            train_exactmatchacc,train_onematchacc,train_loss = sess.run([exactmatch_acc,\
                onematch_acc,total_loss],feed_dict=train_feed_dict)

            # validation set feed dictionary
            val_feed_dict = {labels_list[i]: validation_labels[:,i] for i in range(num_species)}
            val_feed_dict.update({dna: validation_dat, dropout1: 0})

            val_exactmatchacc,val_onematchacc,val_loss = sess.run([exactmatch_acc,\
                onematch_acc,total_loss],feed_dict=val_feed_dict)

            print('\t'.join([str(epoch_num),str(train_exactmatchacc),str(train_onematchacc),\
                str(train_loss),str(val_exactmatchacc),str(val_onematchacc), str(val_loss)]))

            f = open(outfile_name + '.output','a')
            writer = csv.writer(f,delimiter='\t')
            writer.writerow([epoch_num,train_exactmatchacc,train_onematchacc,train_loss,\
                val_exactmatchacc,val_onematchacc,val_loss])
            f.close()

            # save model if current validation loss is lower than the previous lowest
            if val_loss < lowest_val_loss:
                saver.save(sess,outfile_name + '_model')
                lowest_val_loss = val_loss

    # get representational output 
    rep_layer = sess.run(rep,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(outfile_name + '_rep.txt',rep_layer,delimiter='\t')

f.close()


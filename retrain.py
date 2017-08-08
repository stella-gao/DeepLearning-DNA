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

species_dir = 'all10'
model_dir = ''
model_name = 'all8_dense_model'
train_h5file = 'data/h5datasets/all10/train.h5'
validation_h5file = 'data/h5datasets/all10/validation.h5'

outfile_name = 'all10_dense'

# training parameters
epochs = 25
batch_size = 200

# read in HDF5 file & create batch iterator for training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,batch_size)
train_batcher = train_data.batcher()

# read in HDF5 file for validation data
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(10000,validation_file.values()[0].shape[0])
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
labels = graph.get_tensor_by_name("label:0")
dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
rep = graph.get_tensor_by_name('representation/Relu:0')
preds = graph.get_tensor_by_name('dense_1/Softmax:0')
train_step = graph.get_operation_by_name('Adam')

# loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# internal accuracy
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# determine number of total iterations
totalIterations = int(epochs/batch_size*train_size)

with sess.as_default():

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=4)

    lowest_val_loss = 1000

    # initialize writer to write output to file
    f = open(outfile_name + '.output','w')
    writer = csv.writer(f,delimiter='\t')

    print('epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss')
    writer.writerow(['epochs','train_acc','train_loss','loss_acc','val_loss'])

    for i in range(totalIterations):
        batch = train_batcher.next()
        sess.run([train_step],feed_dict={dna: batch['dnaseq'], \
            labels: batch['species_labels'], dropout1: 1})

        # log training and validation accuracy
        if i%1000 == 0:

            epoch_num = i/train_size*batch_size

            train_acc,train_loss = sess.run([accuracy,loss],feed_dict={dna: batch['dnaseq'], \
                labels: batch['species_labels'], dropout1: 0})

            val_acc,val_loss = sess.run([accuracy,loss],feed_dict={dna: validation_dat, \
                labels: validation_labels, dropout1: 0})
            
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
    # save model
    saver.save(sess,outfile_name + '_model_retrain')

    # get representational output 
    rep_layer = sess.run(rep,feed_dict={dna: validation_dat, \
        labels: validation_labels,dropout1: 0})

    # write representational output to file
    np.savetxt(outfile_name + '_rep_retrain.txt',rep_layer,delimiter='\t')


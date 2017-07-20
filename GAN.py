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
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras import callbacks
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import hamming_loss

def insert_motif(seq,motif_list,nummotif_list=False):
	'''takes a DNA sequence and a list of motifs as input and returns the DNA 
	sequence with the motifs randomly inserted (w/o overlap); the number of 
	each motif in the motif_list to be inserted can be specified by an inputted
	list of integers via nummotif_list where each index should correspond to 
	the motif in the motif_list that it enumerates'''

	new_seq = seq
	avail_positions = range(len(seq))

	if not nummotif_list:
		nummotif_list = [1]*len(motif_list)
	for i in range(len(motif_list)):
		for j in range(nummotif_list[i]):
			inserted = False
			while not inserted:
				pos_init = np.random.randint(0,len(seq)-len(motif_list[i]))
				if set(range(pos_init,pos_init+len(motif_list[i]))).issubset(avail_positions):
					inserted = True
					new_seq = new_seq[0:pos_init] + motif_list[i] \
						+ new_seq[pos_init+len(motif_list[i]):]
					avail_positions = [pos for pos in avail_positions if pos not in \
						range(pos_init,pos_init+len(motif_list[i]))]

	return new_seq

def generator(mutated_dna):
	'''builds the generator component of the network'''

	gen_conv1 = Conv2D(num_filters,[filter_height1,filter_width],activation='linear', \
	            kernel_regularizer='l2',padding='same',name='genconv_1')(mutated_dna)
	gen_leak1 = LeakyReLU(alpha=.001)(gen_conv1)
	# gen_pool1 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
	#             name='genAvgPool_1',padding='same')(gen_leak1)
	# # gen_drop1 = Dropout(0.5)(gen_pool1)
	# gen_conv2 = Conv2D(num_filters,[filter_height1,filter_width],activation='linear', \
	#             kernel_regularizer='l2',padding='same',name='genconv_1')(gen_pool1)
	# gen_leak2 = LeakyReLU(alpha=.001)(gen_conv2)
	# gen_pool2 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
	#             name='genAvgPool_1',padding='same')(gen_leak2)
	# upsample1 = UpSampling2D(size=(1,2))
	gen_deconv1 = Conv2DTranspose(num_filters,[filter_height1,filter_width],activation='linear', \
				kernel_regularizer='l2',padding='same',name='gendeconv_1')(gen_leak1)
	gen_leak2 = LeakyReLU(alpha=.001)(gen_deconv1)
	gen_conv2 = Conv2D(1,[filter_height1,filter_width],activation='sigmoid', \
	            kernel_regularizer='l2',padding='same',name='genconv_1')(gen_leak2)

	return gen_conv2

def discriminator(dna):
	'''builds the discriminator component of the network'''

	disc_conv1 = Conv2D(num_filters,[filter_height1,filter_width],activation='linear', \
	            kernel_regularizer='l2',padding='same',name='discconv_1')(dna)
	disc_leak1 = LeakyReLU(alpha=.001)(disc_conv1)
	disc_pool1 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
	            name='discAvgPool_1')(disc_leak1)
	disc_drop1 = Dropout(0.5)(disc_pool1)
	           
	disc_conv2 = Conv2D(num_filters,[filter_height2,filter_width],activation='linear', \
	            kernel_regularizer='l2',padding='same',name='discconv_1')(disc_pool1)
	disc_leak2 = LeakyReLU(alpha=.001)(disc_conv2)
	disc_pool2 = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
	            name='discAvgPool_1')(disc_leak2)
	disc_drop2 = Dropout(0.5)(disc_pool2)

	flat = Flatten()(disc_drop2)
	FC = Dense(50,activation='relu',name='representation')(flat)
	preds = Dense(2,activation='softmax')(FC)

	return preds

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

# define placeholder for DNA sequences (represented via one-hot encoding)
real_dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='real_dna')
mutated_dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1),name='mutated_dna')

# define placeholder for species labels
# labels = tf.placeholder(tf.float32,shape=(None,num_species),name='label')

# build layers of generator component of network
fake_dna = generator(mutated_dna)

discriminator()


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
    saver = tf.train.Saver(max_to_keep=4)

    print('epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss')
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

    # save model
    saver.save(sess,species_dir + '_model')

    # get representational output 
    rep_layer = sess.run(FC,feed_dict={dna: validation_dat, \
        labels: validation_labels,K.learning_phase(): 0})

    # write representational output to file
    np.savetxt(species_dir + '_rep.txt',rep_layer,delimiter='\t')


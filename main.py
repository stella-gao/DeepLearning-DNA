from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from model import *

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, MaxPooling2D, Flatten, Dropout
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras import callbacks
from sklearn.metrics import hamming_loss

species = ['sCer','sPom']
num_species = len(species)

upstream_length = 1000
promoter_length = 500
window_step = 20

seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
    sp in species]
ann_files = ['data/annotation_files/' + sp + '.gtf.gz' for sp in species]
genomedat_dirs = ['data/genome_files/' + sp for sp in species]

# training parameters
epochs = 8
batch_size = 200

# CNN hyperparameters
num_filters = 100
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.001

# read in promoter sequences and convert to one-hot encoding sequences
seq_dicts = [read_fasta_seq(f) for f in seq_files]

# set size of examples for each species to be equal to each other
min_datsize = min([len(seq_dict) for seq_dict in seq_dicts])
subsetIdx = [np.random.choice(range(len(sd)),size=min_datsize,replace=False) \
    for sd in seq_dicts]
subset_regs = [[seq_dicts[i].keys()[j] for j in subsetIdx[i]] for i \
    in range(len(seq_dicts))]
new_seq_dicts = [{reg:seq_dicts[i][reg] for reg in subset_regs[i]} for i \
    in range(len(seq_dicts))]

# use sliding windows to get windows of DNA sequences for each region
print('creating windows...')
windows = [getallWindows(seq_dict,promoter_length,window_step) for seq_dict \
    in new_seq_dicts]
num_windows = [len(windows_set) for windows_set in windows] # windows/species
print('concatenating...')
# arrange sequence data into single array
mat = np.concatenate(windows)
print('training...')
# create species labels
sp_labels = getLabels(num_windows)

# stratify data into training and validation data
allIdx = range(len(mat))
np.random.shuffle(allIdx)
trainIdx = allIdx[0:int(0.8*len(mat))]
valIdx = allIdx[int(0.8*len(mat)):]
train_dat = np.array([mat[i] for i in trainIdx])
train_labels = np.array([sp_labels[i] for i in trainIdx])
validation_dat = np.array([mat[i] for i in valIdx])
validation_labels = np.array([sp_labels[i] for i in valIdx])

# create CallBack Tensorboard object
tbCallBack = callbacks.TensorBoard(log_dir='./sCer_sPom1', histogram_freq=0, \
        write_graph=True, write_images=True)

# build layers of network (using GPU)
model = Sequential()
model.add(Conv2D(num_filters,[filter_height1,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_1', \
            input_shape=(4,promoter_length,1)))
model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
            name='AvgPool_1'))
model.add(Dropout(0.5))
model.add(Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
            kernel_regularizer='l2',padding='valid',name='conv_2'))
model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid',\
            name='AvgPool_2'))
model.add(Dropout(0.5))
# model.add(Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_3'))
# model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid',\
#             name='AvgPool_3'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_species,activation='softmax'))
model.add(Dense(num_species,activation='softmax',name='representation'))

model.compile(loss=categorical_crossentropy,optimizer=Adam(lr=learning_rate),\
            metrics = ['accuracy'])
model.fit(train_dat,train_labels,batch_size=batch_size,epochs=epochs,\
    validation_data=(validation_dat,validation_labels),callbacks=[tbCallBack])

# determine validation accuracy & loss
score = model.evaluate(validation_dat,validation_labels)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# # define placeholder for DNA sequences (represented via one-hot encoding)
# dna = tf.placeholder(tf.float32,shape=(None,4,promoter_length,1))

# # build layers of network
# net = Conv2D(num_filters,[filter_height1,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_1')(dna)
# net = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
#             name='AvgPool_1')(net)
# # net = Dropout(0.25)(net)
# net = Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_2')(net)
# net = AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid',\
#             name='AvgPool_2')(net)
# # net = Dropout(0.25)(net)
# net = Flatten()(net)
# preds = Dense(num_species,activation='softmax',name='representation')(net)

# # define placeholder for species labels
# labels = tf.placeholder(tf.float32,shape=(None,num_species))

# # loss function
# loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# # gradient descent optimizer
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# # initialize all variables
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

# # initialize batcher
# batch = batcher(train_dat.shape[0],batch_size)

# # train the model
# with sess.as_default():
#     for it in range(num_iterations):
#         print(it)
#         for i in range(10):
#             batchIdx = batch.next()
#             train_step.run(feed_dict={dna: train_dat[batchIdx[0]:batchIdx[1]],\
#                 labels: train_labels[batchIdx[0]:batchIdx[1]]})

# # determine validation accuracy
# acc_value = categorical_accuracy(labels,preds)
# with sess.as_default():
#     blah = acc_value.eval(feed_dict={dna: validation_dat, \
#                                     labels: validation_labels})
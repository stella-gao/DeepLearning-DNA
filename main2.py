from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from model import *

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, Flatten, Dropout, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.objectives import binary_crossentropy
from keras.optimizers import SGD
from keras import callbacks

species = ['sCer'] #,'cEleg']
species_list = ['cerevisiae'] #,'elegans']
ontology_terms = ['stress','cell_cycle']

upstream_length = 1000
promoter_length = 500
window_step = 20

seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
    sp in species]
ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]

# training parameters
epochs = 5
batch_size = 20

# CNN hyperparameters
num_filters = 50
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 10
learning_rate = 0.001

# read in promoter sequences and convert to one-hot encoding sequences
seq_dicts = {species_list[i]:read_fasta_seq(seq_files[i]) for i \
    in range(len(species_list))}

# read in GO terms and corresponding genes for each species
GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],species_list) \
    for i in range(len(ontology_files))}

# creates a dictionary of ontologies indexed by species+gene names
geneterm_dict = geneGO_dict(GO_dicts)

# create windows and labels for windows for all genes/regions
(mat,GO_labels) = multilabelWindows(seq_dicts,geneterm_dict,ontology_terms,\
                    promoter_length,window_step)

# print('training...')

# # stratify data into training and validation data
# allIdx = range(len(mat))
# np.random.shuffle(allIdx)
# trainIdx = allIdx[0:int(0.8*len(mat))]
# valIdx = allIdx[int(0.8*len(mat)):]
# train_dat = np.array([mat[i] for i in trainIdx])
# train_labels = np.array([GO_labels[i] for i in trainIdx])
# validation_dat = np.array([mat[i] for i in valIdx])
# validation_labels = np.array([GO_labels[i] for i in valIdx])

# # create CallBack Tensorboard object
# tbCallBack = callbacks.TensorBoard(log_dir='./GraphOntology', histogram_freq=0, \
#         write_graph=True, write_images=True)

# # build layers of network
# model = Sequential()
# model.add(Conv2D(num_filters,[filter_height1,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_1', \
#             input_shape=(4,promoter_length,1)))
# model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
#             name='AvgPool_1'))
# model.add(Dropout(0.5))
# model.add(Conv2D(num_filters,[filter_height2,filter_width],activation='relu', \
#             kernel_regularizer='l2',padding='valid',name='conv_2'))
# model.add(AveragePooling2D((1,pool_size),strides=(1,pool_stride),padding='valid',\
#             name='AvgPool_2'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(num_species,activation='softmax'))
# model.add(Dense(len(ontology_terms),activation='sigmoid',name='representation'))

# # compile model and train
# model.compile(loss=binary_crossentropy,optimizer=SGD(lr=learning_rate),\
#             metrics = ['accuracy'])
# model.fit(train_dat,train_labels,batch_size=batch_size,epochs=epochs,\
#     validation_data=(validation_dat,validation_labels))

# # determine validation accuracy & loss
# score = model.evaluate(validation_dat,validation_labels)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print('Hamming loss:', score[2])


# PATHWAY DATA
# # # read in pathway data
# # pw_dict = pathway_dict(pw_file)

# # # reverse pathway dictionary
# # gene_dict = reverse_pathway_dict(pw_dict)

# # # retain only sequence, pathway data for genes represented in both the sequence
# # # dataset and the pathway dataset
# # gene_dict = {gene:pw_list for gene,pw_list in gene_dict.items() if gene in seq_dict}
# # seq_dict = {gene:seq for gene,seq in seq_dict.items() if gene in gene_dict}

# # # arrange sequence data into single array
# # mat = seqdict2mat(seq_dict)

# # # create pathway labels
# # print('labeling')
# # labels = pathway_labels(seq_dict,gene_dict,pw_dict)
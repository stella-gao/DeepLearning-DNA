import h5py
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Conv2D, AveragePooling2D, Flatten, MaxPooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils.io_utils import HDF5Matrix
from keras.objectives import categorical_crossentropy, kullback_leibler_divergence
from keras import backend as K
from matplotlib import pylab as pl

from MANfunc import *

# species_dir = 'TAL1'
# # read in HDF5 file & create batch iterator for training data
# train_h5file = 'data/simulated_dna/' + str(species_dir) + '/train.h5'
# train_file = h5py.File(train_h5file,'r')
# train_data = TrainData(train_file,batch_size)
# train_batcher = train_data.batcher()

realdna_h5file = 'data/simulated_dna/TAL1.h5'
f = h5py.File(realdna_h5file,'r')
dnaseq_dat = f['dnaseq']
train_size = dnaseq_dat.shape[0]

# training parameters
epochs = 30
batch_size = 50
disc2mut_ratio = 20

# CNN hyperparameters
num_filters = 50
pool_size = 2
pool_stride = 2
filter_height1 = 4
filter_height2 = 1
filter_width = 5
learning_rate = 0.0001

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

# determine number of total iterations
totalIterations = int(float(train_size)*epochs/batch_size)

inp = tf.placeholder(tf.float32, [None, 4, dnaseq_dat.shape[2], 1], name='dna_input')
target = tf.placeholder(tf.float32, [None, 2], name='label')
# temperature
tau = tf.constant(1.0, name='temperature')

# with tf.variable_scope('mutator'):    
#     cn1 = Conv2D(64, [4, 5],
#                     activation='relu',
#                     kernel_regularizer='l2',
#                     padding='valid',
#                     name='conv_1')

#     cn2 = Conv2D(32, [1, 5],
#                     activation='relu',
#                     kernel_regularizer='l2',
#                     padding='valid',
#                     name='conv_2')
#     mp1 = MaxPooling2D((1, 2), strides=(1, 2))
#     cn3 = Conv2D(16, [1, 5],
#                     activation='relu',
#                     kernel_regularizer='l2',
#                     padding='valid',
#                     name='conv_3')
#     mp2 = MaxPooling2D((1, 2), strides=(1, 2))
#     fl1 = Flatten()
#     dn1 = Dense(50, name='representation', activation='tanh')

#     dn2m = Dense(dnaseq_dat.shape[2], name='prediction', activation='linear')
#     mutant_repr = dn1(fl1(mp2(cn3(mp1(cn2(cn1(inp)))))))
#     logits  =dn2m(mutant_repr)
#     logits_dna = tf.reshape(logits, [-1, 1, dnaseq_dat.shape[2], 1])
#     # sample and reshape back (shape=(batch_size,N,K))
#     # set hard=True for ST Gumbel-Softmax
#     mutant = gumbel_softmax(inp, logits_dna, tau, hard=False)
    
with tf.variable_scope('mutator'):    

	# # transition layers
	# trans1_conv = Conv2D(num_filters,[filter_height1,filter_width],activation='linear', \
	#             kernel_regularizer='l2',padding='valid',name='trans1_conv')(inp)
	# trans1_leak = LeakyReLU(alpha=.001)(trans1_conv)
	# trans1_pool = AveragePooling2D((1,pool_size),strides=(1,pool_stride),\
	#                 name='AvgPool_trans1')(trans1_leak)
	# trans1_drop = Dropout(0.5)(trans1_conv)

	# # dense block
	# denseblock1 = dense_block(trans1_drop,1,num_filters=20)

	drop1 = conv_pool_drop(inp,1,filter_height1,filter_width)
	drop2 = conv_pool_drop(drop1,2,filter_height2,filter_width)
	drop3 = conv_pool_drop(drop2,3,filter_height2,filter_width)

	fl1 = Flatten()(drop3)
	mutant_repr = Dense(50, name='mutant_representation', activation='tanh')(fl1)

	logits = Dense(dnaseq_dat.shape[2], name='prediction', \
    	activation='linear')(mutant_repr)

   	logits_dna = tf.reshape(logits, [-1, 1, dnaseq_dat.shape[2], 1])

    # sample and reshape back (shape=(batch_size,N,K))
    # set hard=True for ST Gumbel-Softmax	
	mutant = gumbel_softmax(inp, logits_dna, tau, hard=True)

with tf.variable_scope('discriminator'):
    cn1 = Conv2D(64, [4, 5],
                    activation='relu',
                    kernel_regularizer='l2',
                    padding='valid',
                    name='conv_1')

    cn2 = Conv2D(32, [1, 5],
                    activation='relu',
                    kernel_regularizer='l2',
                    padding='valid',
                    name='conv_2')
    mp1 = MaxPooling2D((1, 2), strides=(1, 2))
    cn3 = Conv2D(16, [1, 5],
                    activation='relu',
                    kernel_regularizer='l2',
                    padding='valid',
                    name='conv_3')
    mp2 = MaxPooling2D((1, 2), strides=(1, 2))
    fl1 = Flatten()
    dn1 = Dense(50, name='representation', activation='tanh')

    dn2d = Dense(1, name='prediction', activation='sigmoid')

    # mutant probability
    p_mutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(mutant))))))))

    # non-mutant/real probability
    p_nonmutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(inp))))))))

    # features branch
    featConv = Conv2D(64, [4, 5],activation='relu',kernel_regularizer='l2',
                    padding='valid',name='featuresConv')
    featFlat = Flatten()
    featDense = Dense(50,name='features',activation='softmax')

    features_mutant = featDense(featFlat(featConv(mutant))) # mutant features
    features_nonmutant = featDense(featFlat(featConv(inp))) # non-mutant features

disc_acc = (tf.reduce_mean(1-p_mutant) + tf.reduce_mean(p_nonmutant))/2

mutator_loss = tf.reduce_mean(inp*tf.log(K.clip(mutant, K.epsilon(), 1-K.epsilon())))

# sum of squared errors between feature maps of mutant and non-mutant inputs
# featurematch_loss = tf.reduce_sum(tf.square(tf.subtract(features_nonmutant,features_mutant)))

# K-L divergence between feature maps of mutant and non-mutant inputs
featurematch_loss = tf.reduce_mean(kullback_leibler_divergence(features_nonmutant,features_mutant))

D_loss = - tf.reduce_mean(tf.log(p_nonmutant)) - tf.reduce_mean(tf.log(1-p_mutant))
M_loss = featurematch_loss # -tf.reduce_mean(tf.log(p_mutant)) #+ mutator_loss

# gradient clipping
# theta_D is list of D's params
theta_D = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
           if 'discriminator' in x.name]
theta_M = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
           if 'mutator' in x.name]
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

D_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
            .minimize(D_loss, var_list=theta_D))
M_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
            .minimize(M_loss, var_list=theta_M))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batcher = get_batch(dnaseq_dat, batch_size=batch_size)
temperature = 1.

# for it in range(int(totalIterations/float(disc2mut_ratio))):
#     for _ in range(disc2mut_ratio):
#         xt= batcher.next()
#         _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], \
#                                      feed_dict={inp: xt, tau: temperature,K.learning_phase(): 1})
#     _, M_loss_curr, acc, np_mutloss = sess.run([M_solver, M_loss, disc_acc, mutator_loss], feed_dict={inp: xt, tau: temperature, K.learning_phase(): 1})
#     # blah = xt
#     # boo = sess.run(mutant,feed_dict={inp: xt, tau: temperature,K.learning_phase(): 0})
#     # print(boo)
#     if it % 100 == 0:

#     	epoch_num = it*float(disc2mut_ratio)/train_size*batch_size
#     	print('{} Mutator loss: {} \t and mut loss: {}'.format(epoch_num, M_loss_curr, np_mutloss))
#     	print('{} Discriminator loss: {} \t and accuracy: {}'.format(epoch_num, D_loss_curr, acc))
#     	print('\n')
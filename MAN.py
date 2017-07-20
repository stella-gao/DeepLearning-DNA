import h5py
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Conv2D, AveragePooling2D, Flatten, MaxPooling2D, Dropout
from keras.utils.io_utils import HDF5Matrix
from keras.objectives import categorical_crossentropy
from keras import backend as K
from matplotlib import pylab as pl

from MANfunc import *

realdna_5hfile = 'data/simulated_dna/TAL1.h5'
f = h5py.File(realdna_5hfile,'r')
dnaseq_dat = f['dnaseq']
train_size = dnaseq_dat.shape[0]

# training parameters
epochs = 15
batch_size = 200
disc2mut_ratio = 1

# determine number of total iterations
totalIterations = int(float(train_size)*epochs/batch_size)

inp = tf.placeholder(tf.float32, [None, 4, dnaseq_dat.shape[2], 1], name='dna_input')
target = tf.placeholder(tf.float32, [None, 2], name='label')
# temperature
tau = tf.constant(1.0, name='temperature')

with tf.variable_scope('mutator'):    
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

    dn2m = Dense(dnaseq_dat.shape[2], name='prediction', activation='linear')
    mutant_repr = dn1(fl1(mp2(cn3(mp1(cn2(cn1(inp)))))))
    logits  =dn2m(mutant_repr)
    logits_dna = tf.reshape(logits, [-1, 1, dnaseq_dat.shape[2], 1])
    # sample and reshape back (shape=(batch_size,N,K))
    # set hard=True for ST Gumbel-Softmax
    mutant = gumbel_softmax(inp, logits_dna, tau, hard=False)
    
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
    p_mutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(mutant))))))))
    p_nonmutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(inp))))))))
    
disc_acc = (tf.reduce_mean(1-p_mutant) + tf.reduce_mean(p_nonmutant))/2

mutator_loss = tf.reduce_mean(inp*tf.log(K.clip(mutant, K.epsilon(), 1-K.epsilon())))

D_loss = - tf.reduce_mean(tf.log(p_nonmutant)) - tf.reduce_mean(tf.log(1-p_mutant))
M_loss = -tf.reduce_mean(tf.log(p_mutant)) #+ mutator_loss

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

batcher = get_batch(dnaseq_dat, batch_size=50)
temperature = 1.
for it in range(int(totalIterations/float(disc2mut_ratio))):
    for _ in range(disc2mut_ratio):
        xt= batcher.next()
        _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], \
                                     feed_dict={inp: xt, tau: temperature})
    _, M_loss_curr, acc, np_mutloss = sess.run([M_solver, M_loss, disc_acc, mutator_loss], feed_dict={inp: xt, tau: temperature})
    
    if it % 100 == 0:
    	epoch_num = it*float(disc2mut_ratio)/train_size*batch_size
    	print('{} Mutator loss: {} \t and mut loss: {}'.format(epoch_num, M_loss_curr, np_mutloss))
    	print('{} Discriminator loss: {} \t and accuracy: {}'.format(epoch_num, D_loss_curr, acc))
    	print('\n')
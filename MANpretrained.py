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
from model import *

species_dir = 'TAL1_background'
train_h5file = 'data/simulated_dna/' + str(species_dir) + '/train.h5'
validation_h5file = 'data/simulated_dna/' + str(species_dir) + '/validation.h5'
background_h5file = 'data/simulated_dna/background.h5'

# Discriminator training parameters
Disc_epochs = 2
Disc_batch_size = 50

# Mutator training parameters
Mut_epochs = 250
Mut_batch_size = 50

# read in HDF5 file & create batch iterator for Discriminator training data
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,Disc_batch_size)
Disc_batcher = train_data.batcher()
Disc_train_size = train_file.values()[0].shape[0]

# read in HDF5 file for Discriminator validation data 
validation_file = h5py.File(validation_h5file,'r')
val_datsize = min(500,validation_file.values()[0].shape[0])
validation_dat = validation_file['dnaseq'][0:val_datsize]
validation_labels = validation_file['labels'][0:val_datsize]

# read in HDF5 file & create batch iterator for Mutator input data
Mut_file = h5py.File(background_h5file,'r')
Mut_data = TrainData(Mut_file,Mut_batch_size)
Mut_batcher = Mut_data.batcher()
Mut_train_size = Mut_file.values()[0].shape[0]

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

# define relevant placeholders
inp = tf.placeholder(tf.float32, [None, 4, train_file['dnaseq'].shape[2], 1], name='dna_input')
labels = tf.placeholder(tf.float32, [None, 2], name='label')
tau = tf.constant(1.0, name='temperature') # temperature
    
with tf.variable_scope('mutator'):    

    drop1 = conv_drop(inp,1,filter_height1,filter_width)
    drop2 = conv_drop(drop1,2,filter_height2,filter_width)
    drop3 = conv_pool_drop(drop2,3,filter_height2,filter_width)

    fl1 = Flatten()(drop3)
    mutant_repr = Dense(50, name='mutant_representation', activation='tanh')(fl1)

    logits = Dense(train_file['dnaseq'].shape[2], name='prediction', \
        activation='linear')(mutant_repr)

    logits_dna = tf.reshape(logits, [-1, 1, train_file['dnaseq'].shape[2], 1])

    # sample and reshape back (shape=(batch_size,N,K))
    # set hard=True for ST Gumbel-Softmax   
    mutant = gumbel_softmax(inp, logits_dna, tau, hard=False)

    # create copy of mutant (for accessing later)
    mutant_copy = tf.identity(mutant,name='mutant')

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
    mp1 = AveragePooling2D((1, 2), strides=(1, 2))
    cn3 = Conv2D(16, [1, 5],
                    activation='relu',
                    kernel_regularizer='l2',
                    padding='valid',
                    name='conv_3')
    mp2 = AveragePooling2D((1, 2), strides=(1, 2))
    fl1 = Flatten()
    dn1 = Dense(50, name='representation', activation='tanh')

    dn2d = Dense(2, name='prediction', activation='softmax')

    # mutant probability (of being real)
    p_mutant = dn2d(dn1(fl1(mp1(cn1(mutant)))))
    # p_mutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(mutant))))))))

    # non-mutant/real probability (of being real)
    p_nonmutant = dn2d(dn1(fl1(mp1(cn1(inp)))))
    # p_nonmutant = dn2d(dn1(fl1(mp2(cn3(mp1(cn2(cn1(inp))))))))

    # create copies of mutant and non-mutant predictions (for accessing later)
    p_mutant_copy = tf.identity(p_mutant,name='prediction_mutant')
    p_nonmutant_copy = tf.identity(p_nonmutant,name='prediction_nonmutant')

    # features branch
    featConv = Conv2D(64, [4, 5],activation='relu',kernel_regularizer='l2',
                    padding='valid',name='featuresConv')
    featFlat = Flatten()
    featDense = Dense(50,name='features',activation='softmax')

    # features_mutant = featDense(featFlat(featConv(mutant))) # mutant features
    # features_nonmutant = featDense(featFlat(featConv(inp))) # non-mutant features

disc_acc = (tf.reduce_mean(1-p_mutant) + tf.reduce_mean(p_nonmutant))/2

mutator_loss = tf.reduce_mean(inp*tf.log(K.clip(mutant, K.epsilon(), 1-K.epsilon())))

# sum of squared errors between feature maps of mutant and non-mutant inputs
# featurematch_loss = tf.reduce_sum(tf.square(tf.subtract(features_nonmutant,features_mutant)))

# K-L divergence between feature maps of mutant and non-mutant inputs
# featurematch_loss = tf.reduce_mean(kullback_leibler_divergence(features_nonmutant,features_mutant))

# mutant change
mutant_change = tf.reduce_sum(tf.square(tf.subtract(inp,mutant)))

D_loss = tf.reduce_mean(categorical_crossentropy(labels, p_nonmutant))
# D_loss = - tf.reduce_mean(tf.log(p_nonmutant)) - tf.reduce_mean(tf.log(1-p_mutant))
M_loss = tf.reduce_mean(categorical_crossentropy(labels,p_mutant)) - mutator_loss #-tf.reduce_mean(tf.log(p_mutant[:,0]))

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

# internal accuracy
nonmutant_correct_pred = tf.equal(tf.argmax(p_nonmutant, 1), tf.argmax(labels, 1))
nonmutant_accuracy = tf.reduce_mean(tf.cast(nonmutant_correct_pred, tf.float32))

mutant_correct_pred = tf.equal(tf.argmax(p_mutant, 1), tf.argmax(labels, 1))
mutant_accuracy = tf.reduce_mean(tf.cast(mutant_correct_pred, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# determine number of total iterations to train Discriminator
DiscTotalIterations = int(float(Disc_train_size)*Disc_epochs/Disc_batch_size)
print(DiscTotalIterations)

# determine number of total iterations to train Mutator
MutTotalIterations = int(float(Mut_train_size)*Mut_epochs/Mut_batch_size)
print(MutTotalIterations)

with sess.as_default():

    # print('epochs\ttrain_acc\ttrain_loss\tval_acc\tval_loss')

    # # train Discriminator first
    # for i in range(DiscTotalIterations):
    #     Disc_batch = Disc_batcher.next()
    #     sess.run([D_solver],feed_dict={inp: batch['dnaseq'], \
    #         labels: batch['labels'], K.learning_phase(): 1})

    #     # log training and validation accuracy
    #     if i%800 == 0:

    #         epoch_num = float(i)/Disc_train_size*Disc_batch_size

    #         train_acc,train_loss = sess.run([nonmutant_accuracy,D_loss],feed_dict={inp: batch['dnaseq'], \
    #             labels: batch['labels'], K.learning_phase(): 0})

    #         val_acc,val_loss = sess.run([nonmutant_accuracy,D_loss],feed_dict={inp: validation_dat, \
    #             labels: validation_labels, K.learning_phase(): 0})

    #         print('\t'.join([str(epoch_num),str(train_acc),str(train_loss), \
    #             str(val_acc),str(val_loss)]))

    # print(sess.run([p_nonmutant],feed_dict={inp: validation_dat,labels:validation_labels,K.learning_phase():0}))

    # print('Finished training Discriminator.')
    # print('--------------------')
    # print('Training Mutator...')
    # print('epochs\tMutator_loss\tMutator_accuracy\tMutant_change')

    # initialize model saver
    saver = tf.train.Saver(max_to_keep=2)

    # initialize writer to write output to file
    f = open('MAN_' + species_dir + '.output','w')
    writer = csv.writer(f,delimiter='\t')

    # train Mutator
    temperature = 1.
    lowest_Mut_loss = 1000
    for i in range(MutTotalIterations):
        Mut_batch = Mut_batcher.next()
        M_acc_val,_ = sess.run([mutant_accuracy,M_solver],feed_dict={inp: Mut_batch['dnaseq'],labels: np.array([[1.,0.]]), \
            tau: temperature, K.learning_phase(): 1})

        # train Discriminator
        if i%3000 == 0:
            Disc_batch = Disc_batcher.next()
            sess.run([D_solver],feed_dict={inp: Disc_batch['dnaseq'], \
                labels: Disc_batch['labels'], K.learning_phase(): 1})

            # determine number of epochs trained for Mutator, Discriminator
            Mut_epoch_num = float(i)/Mut_train_size*Mut_batch_size
            Disc_epoch_num = float(i)/Disc_train_size*Disc_batch_size/3000.

            # get Discriminator statistics
            train_acc,train_loss = sess.run([nonmutant_accuracy,D_loss],feed_dict={inp: Disc_batch['dnaseq'], \
                labels: Disc_batch['labels'], K.learning_phase(): 0})

            val_acc,val_loss = sess.run([nonmutant_accuracy,D_loss],feed_dict={inp: validation_dat, \
                labels: validation_labels, K.learning_phase(): 0})

            print('Discriminator Statistics:')
            print('Discriminator Epochs: ' + str(Disc_epoch_num))
            print('Training Accuracy: ' + str(train_acc))
            print('Training Loss: ' + str(train_loss))
            print('Validation Accuracy: ' + str(val_acc))
            print('Validation Loss: ' + str(val_loss))
            print('--------------------')

            writer.writerow(['Discriminator Statistics:'])
            writer.writerow(['Discriminator Epochs:',Disc_epoch_num])
            writer.writerow(['Training Accuracy:',train_acc])
            writer.writerow(['Training Loss:',train_loss])
            writer.writerow(['Validation Accuracy:',val_acc])
            writer.writerow(['Validation Loss:',val_loss])
            writer.writerow(['--------------------'])

            # get Mutator statistics
            # print(sess.run(p_mutant,feed_dict={inp: Mut_batch['dnaseq'],K.learning_phase(): 0}))
            M_acc_val,M_loss_val,M_change_val = sess.run([mutant_accuracy,M_loss,mutant_change],feed_dict={inp: Mut_batch['dnaseq'], \
                labels: np.array([[1.,0.]]), tau: temperature, K.learning_phase(): 0})
            # print(str(epoch_num)+'\t'+str(M_loss_val) + '\t' + str(M_acc_val) + '\t' + str(M_change_val))

            print('Mutator Statistics:')
            print('Mutator Epochs: ' + str(Mut_epoch_num))
            print('Mutator Loss: ' + str(M_loss_val))
            print('Mutator Accuracy: ' + str(M_acc_val))
            print('Mutator Change: ' + str(M_change_val))
            print('\n\n')

            writer.writerow(['Mutator Statistics:'])
            writer.writerow(['Mutator Epochs:',Mut_epoch_num])
            writer.writerow(['Mutator Loss:',M_loss_val])
            writer.writerow(['Mutator Accuracy:',M_acc_val])
            writer.writerow(['Mutator Change:',M_change_val])
            writer.writerow(['\n\n'])

            # save model if current validation loss is lower than the previous lowest
            if M_loss_val < lowest_Mut_loss:
                saver.save(sess,species_dir + '_Mutmodel')
                lowest_val_loss = M_loss_val

        # if i%5000 == 0:
        #     print('---------------------')
        #     print('Checking to see if Discriminator changes...')
        #     val_acc,val_loss,prob_vals = sess.run([nonmutant_accuracy,D_loss,p_nonmutant],feed_dict={inp: validation_dat, \
        #         labels: validation_labels, K.learning_phase(): 0})
        #     prob_vals = sess.run(p_nonmutant,feed_dict={inp: validation_dat, \
        #         labels: validation_labels, K.learning_phase(): 0})
        #     print('Validation Accuracy: ' + str(val_acc))
        #     print('Validation Loss: ' + str(val_loss))
        #     print(prob_vals)
# batcher = get_batch(dnaseq_dat, batch_size=batch_size)
# 
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

#         epoch_num = it*float(disc2mut_ratio)/train_size*batch_size
#         print('{} Mutator loss: {} \t and mut loss: {}'.format(epoch_num, M_loss_curr, np_mutloss))
#         print('{} Discriminator loss: {} \t and accuracy: {}'.format(epoch_num, D_loss_curr, acc))
#         print('\n')
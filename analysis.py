import csv
import numpy as np
import h5py

import tensorflow as tf

from sklearn.mixture import GaussianMixture

def write_metadata(label_set,data_dir,file_name):
	'''writes metadata info (labels) to .tsv file - to be used in visualization
	in Tensorflow Projector'''

	f = h5py.File('data/h5datasets/' + data_dir + '/' + file_name,'r')
	g = open(data_dir + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')

	writer.writerow(['index','label'])
	if file_name == 'validation.h5':
		val_datsize = min(10000,f['labels'].shape[0])
	else:
		val_datsize = f['labels'].shape[0]
	for i in range(val_datsize):
		labelIdx = list(f['labels'][i]).index(max(f['labels'][i]))
		writer.writerow([labelIdx,label_set[labelIdx]])

	f.close()
	g.close()

def write_metadataGO(label_set,data_dir):
	'''writes metadata info (labels) to .tsv file for GO prediction - to be 
	used in visualization in Tensorflow Projector 
	NOTE: last element of label set should be label for examples with more
	than one label, i.e. "both", "multiple" '''

	f = h5py.File('data/h5datasets_GO/' + data_dir + '/validation.h5','r')
	g = open(data_dir + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')

	writer.writerow(['index','label'])
	val_datsize = min(10000,f['labels'].shape[0])
	for i in range(val_datsize):
		if sum(f['labels'][i]) >= 2:
			labelIdx = len(label_set)-1
		else:
			labelIdx = list(f['labels'][i]).index(max(f['labels'][i]))
		writer.writerow([labelIdx,label_set[labelIdx]])

	f.close()
	g.close()

def get_representations(model_name,model_dir,testdata_file,write2file=False):
	'''restores a pre-trained model stored in the inputted directory and tests
	it on a set of test data given by an inputted H5 file name, returning a 
	numpy array of the representational layer outputs'''

	# restore graph
	sess = tf.Session()
	saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
	saver.restore(sess,'./' + model_dir + model_name)

	graph = tf.get_default_graph()

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	labels = graph.get_tensor_by_name("label:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	opt = graph.get_tensor_by_name('representation/Relu:0')

	# load test data
	f = h5py.File(testdata_file,'r')
	test_dat = f['dnaseq'][:]
	test_labels = f['labels'][:]
	f.close()

	# run session and get output of representational layer
	feed_dict = {dna: test_dat, labels: test_labels, dropout1: 0}
	rep_output = sess.run(opt,feed_dict)

	if write2file:
		np.savetxt(model_name + '_rep.txt',rep_output,delimiter='\t')

	else:
		return rep_output

def filterLabels(representation_file,h5_file,label_filter):
	'''takes in a .txt file of representational layer outputs along with the
	corresponding raw data file (in H5 format) and filters out the 
	representational layer outputs that are associated with an inputted label'''

	f = h5py.File(h5_file,'r')

	f.close()

model_name = 'sCer_cEleg_Mouse_Human_model'
model_dir = 'results/saved_models/sCer_cEleg_Mouse_Human/'
testdata_file = 'data/h5datasets/sCer_cEleg_Mouse_Human/all.h5'

# a = get_representations(model_name,model_dir,testdata_file,True)

# file_name = 'all.h5'
# write_metadata(['sCer','sPom'],'sCer_sPom',file_name)
# d = np.loadtxt('sCer_sPom_rep.txt',delimiter='\t')


write_metadata(['sCer','cEleg','Mouse','Human'],'sCer_cEleg_Mouse_Human','all.h5')
# write_metadataGO(['stress','cell_cycle','chrorg','multiple'],'sCer_stress+cc+chrorg')

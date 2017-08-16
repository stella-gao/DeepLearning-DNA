import csv
import numpy as np
import h5py
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow as tf
from keras.models import model_from_json
from model import *

### METADATA ##################################################################

def write_metadata(label_set,data_dir,file_name):
	'''writes metadata info (labels) to .tsv file - to be used in visualization
	in Tensorflow Projector'''

	f = h5py.File('data/h5datasets/' + data_dir + '/' + file_name,'r')
	g = open(data_dir + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')

	writer.writerow(['index','species_label','gene_names'])
	if file_name == 'validation.h5':
		val_datsize = min(10000,f['species_labels'].shape[0])
	else:
		val_datsize = f['species_labels'].shape[0]
	for i in range(val_datsize):
		labelIdx = list(f['species_labels'][i]).index(max(f['species_labels'][i]))
		gene_name = str(f['genes'][i])
		writer.writerow([labelIdx,label_set[labelIdx],gene_name])

	f.close()
	g.close()

def write_metadataGO(label_set,data_dir,multiple=True):
	'''writes metadata info (labels) to .tsv file for GO prediction - to be 
	used in visualization in Tensorflow Projector
	NOTE: option to write labels as true combination of labels or as just 
	"multiple" to delineate multi-labels'''

	f = h5py.File('data/h5datasets_GO/' + data_dir + '/validation.h5','r')
	g = open(data_dir + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')

	writer.writerow(['index','label'])
	val_datsize = min(10000,f['labels'].shape[0])
	for i in range(val_datsize):

		if multiple:
			if sum(f['labels'][i]) >= 2:
				labelIdx = 'NA'
				writer.writerow([labelIdx,'multiple'])
			else:
				labelIdx = list(f['labels'][i]).index(max(f['labels'][i]))
				writer.writerow([labelIdx,label_set[labelIdx]])

		else:
			metadata_Idxs = []
			metadata_labels = []
			for j in range(f['labels'].shape[1]):
				if f['labels'][i][j] == 1:
					metadata_Idxs.append(str(j))
					metadata_labels.append(label_set[j])
			writer.writerow(['+'.join(metadata_Idxs),'+'.join(metadata_labels)])

	f.close()
	g.close()

def filterLabels(representation_file,metadata_file,index_set,output_file):
	'''takes in a .txt file of representational layer outputs along with the
	corresponding raw data file (in H5 format) and filters out the 
	representational layer outputs that are associated with an inputted label'''

	rep_dat = np.loadtxt(representation_file,delimiter='\t')

	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')

	g = open(output_file + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')
	
	i = 0
	index_list = []
	writer.writerow(reader.next()) # skip metadata file header
	for line in reader:
		if int(line[0]) in index_set:
			writer.writerow(line)
			index_list.append(i)
		i += 1

	np.savetxt(output_file + '_filteredrep.txt',rep_dat[np.array(index_list)], \
		delimiter='\t')

	f.close()
	g.close()

def filterGenes(representation_file,metadata_file,index_set,gene_list,output_file):
	'''takes in a .txt file of representational layer outputs along with the
	corresponding raw data file (in H5 format) and filters out the 
	representational layer outputs that are associated with an inputted gene list'''	

	rep_dat = np.loadtxt(representation_file,delimiter='\t')

	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')

	g = open(output_file + '_metadata.tsv','w')
	writer = csv.writer(g,delimiter='\t')
	
	i = 0
	index_list = []
	writer.writerow(reader.next()) # skip metadata file header
	for line in reader:
		if int(line[0]) in index_set and line[2].lower() in gene_list:
			writer.writerow(line)
			index_list.append(i)
		i += 1

	np.savetxt(output_file + '_filteredrep.txt',rep_dat[np.array(index_list)], \
		delimiter='\t')

	f.close()
	g.close()

def filterSpeciesMetadataOnly(metadata_file,index_set,output_file):
	'''takes in a metadata file and returns a filtered version consisting of
	only species referenced by the index_set list parameter'''

	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')
	reader.next()

	g = open(output_file,'w')

	for line in reader:
		if int(line[0]) in index_set:
			g.write(line[2] + '\n')

	f.close()
	g.close()

### TENSORFLOW PROJECTOR ######################################################

def getSelectedPointsTFProj(state_file,metadata_file,filter_species=False,write2file=False):
	'''takes as input a "state" file (generated from Tensorflow Projector) in
	JSON format as well as the accompanying metadata file for the projected data
	and returns a list of gene names associated with the selected points

	NOTE: input a species name for the filter_species parameter to filter out
	the genes in the selected list of genes that belong to the specified species'''

	import json

	# get list of selected point indices
	f = open(state_file,'r')
	data = json.load(f)
	f.close()
	dataIdx_list = data[0]['filteredPoints']

	# get list of gene names
	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')
	reader.next() # skip header
	species_gene_list = [(line[1],line[2]) for line in reader]
	f.close()

	if not filter_species:
		gene_list = [species_gene_list[i][1] for i in dataIdx_list]
	else:
		gene_list = [species_gene_list[i][1] for i in dataIdx_list \
			if species_gene_list[i][0] == filter_species]

	if write2file:
		f = open('selectedGenes.txt','w')
		[f.write(gene + '\n') for gene in gene_list]
		f.close()

	return gene_list

def getAllProjectedPoints(state_file,ndim=3,write2file=False):
	'''takes as input a "state" file (generated from Tensorflow Projector) in
	JSON format as well as the accompanying metadata file for the projected data
	and returns an array of projected points alongside the corresponding gene names

	NOTE: input a species name for the filter_species parameter to filter out
	the genes in the selected list of genes that belong to the specified species'''	

	import json

	# get list of selected projections
	f = open(state_file,'r')
	data = json.load(f)[0]
	f.close()
	data_list = data['projections']

	# get n dimensions of projections
	projectedPoints = np.array([np.array([data_list[i]['pca-'+str(j)] \
		for j in range(ndim)]) for i in range(len(data_list))])

	if write2file:
		np.savetxt('projectedPoints.txt',projectedPoints,delimiter='\t')

	return projectedPoints	

### GO TERM DATA ##############################################################

def getGOcounts(infile,species):
	'''returns a list of tuples containing GO terms associated with the inputted 
	species and their respective counts'''

	from collections import Counter
	import operator

	f = open(infile,'r')
	reader = csv.reader(f,delimiter='\t')

	GOterm_list = []
	for line in reader:
		if species in line[1]:
			GOterm_list.append(line[3])

	counts = Counter(GOterm_list)

	return sorted(counts.items(), key=operator.itemgetter(1))

### NN LAYER OUTPUTS #########################################################

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
	rep = graph.get_tensor_by_name('representation/Relu:0')
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# load test data
	f = h5py.File(testdata_file,'r')
	test_dat = f['dnaseq'][:]
	test_labels = f['species_labels'][:]
	f.close()

	# internal accuracy
	correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	if write2file:
		for i in range(0,test_dat.shape[0],50):
			print(i)
			# run session and get output of representational layer
			feed_dict = {dna: test_dat[i:i+50], labels: test_labels[i:i+50], dropout1: 0}
			rep_output = sess.run(rep,feed_dict)

			f = open(model_name + '_rep.txt','a')
			writer = csv.writer(f,delimiter='\t')
			for j in range(rep_output.shape[0]):
				writer.writerow(rep_output[j])
			f.close()

	else:
		feed_dict = {dna: test_dat, labels: test_labels, dropout1: 0}
		rep_output = sess.run(rep,feed_dict)
		return rep_output

def getPredictions(model_name,model_dir,testdata_file,label_names,limit=10000,write2file=False):
	'''restores a pre-trained model stored in the inputted directory and tests
	it on a set of test data given by an inputted H5 file name, returning the
	raw predictions, labeled raw predictions, and confusion matrix'''

	from sklearn.metrics import confusion_matrix

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
	# preds = graph.get_tensor_by_name('dense_1/Softmax:0')
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# load test data
	f = h5py.File(testdata_file,'r')
	if limit:
		test_dat = f['dnaseq'][0:limit]
		test_labels = f['species_labels'][0:limit]
	else:
		test_dat = f['dnaseq'][:]
		test_labels = f['species_labels'][:]
	f.close()

	prediction_names = []
	if write2file:
		for i in range(0,test_dat.shape[0],50):
			# run session and get output of representational layer
			feed_dict = {dna: test_dat[i:i+50], labels: test_labels[i:i+50], dropout1: 0}
			pred_probs = sess.run(preds,feed_dict)

			f = open(model_name + '_predProbs.txt','a')
			writer = csv.writer(f,delimiter='\t')
			for j in range(pred_probs.shape[0]):
				writer.writerow(pred_probs[j])
			f.close()

			prediction_names.extend([label_names[np.argmax(pred_probs[i])] \
				for i in range(pred_probs.shape[0])])
	else:
		feed_dict = {dna: test_dat, labels: test_labels, dropout1: 0}
		pred_probs = sess.run(preds,feed_dict)

	# truelabel_names = f['species'][0:10000]
	# truelabel_names = [str(i) for i in truelabel_names]
	truelabel_names = [label_names[np.argmax(label)] for label in test_labels]
	prediction_names = [label_names[np.argmax(pred)] for pred in pred_probs]

	return pred_probs,confusion_matrix(truelabel_names,prediction_names,\
		labels=label_names)

def get_representationsBinary(model_name,model_dir,testdata_file,write2file=False):
	'''restores a pre-trained model stored in the inputted directory and tests
	it on a set of test data given by an inputted H5 file name, returning a 
	numpy array of the representational layer outputs (for multi-label binary 
	classification problems'''

	# load test data
	f = h5py.File(testdata_file,'r')
	test_dat = f['dnaseq'][:]
	test_labels = f['species_labels'][:]
	f.close()

	# restore graph
	sess = tf.Session()
	saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
	saver.restore(sess,'./' + model_dir + model_name)

	graph = tf.get_default_graph()

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	rep = graph.get_tensor_by_name('representationAll/Relu:0')
	preds_list = [graph.get_tensor_by_name('dense_'+str(i+1)+'/Softmax:0') for \
		i in range(test_labels.shape[1])]
	train_step = graph.get_operation_by_name('Adam')
	# labels_list = [graph.get_tensor_by_name("label" + str(i) + ":0") for i \
	# 	in range(test_labels.shape[1])]

	labels_list = [graph.get_tensor_by_name("Placeholder:0")]
	labels_list.extend([graph.get_tensor_by_name("Placeholder_"+str(i)+":0") for \
		i in range(1,test_labels.shape[1])])

	if write2file:
		for i in range(0,test_dat.shape[0],50):
			print(i)
			# run session and get output of representational layer
			feed_dict = {labels_list[j]: test_labels[i:i+50,j] for j in range(test_labels.shape[1])}
			feed_dict.update({dna: test_dat[i:i+50], dropout1: 0})
			rep_output = sess.run(rep,feed_dict)

			f = open(model_name + '_rep.txt','a')
			writer = csv.writer(f,delimiter='\t')
			for j in range(rep_output.shape[0]):
				writer.writerow(rep_output[j])
			f.close()
	else:
		feed_dict = {labels_list[j]: test_labels[:,j] for j in range(test_labels.shape[1])}
		feed_dict.update({dna: test_dat, dropout1: 0})
		rep_output = sess.run(rep,feed_dict)
		return rep_output

def getPredictionsBinary(model_name,model_dir,testdata_file,limit=10000,write2file=False):
	'''restores a pre-trained model stored in the inputted directory and tests
	it on a set of test data given by an inputted H5 file name, returning the
	raw predictions'''

	# load test data
	f = h5py.File(testdata_file,'r')
	if limit:
		test_dat = f['dnaseq'][0:limit]
		test_labels = f['species_labels'][0:limit]
	else:
		test_dat = f['dnaseq'][:]
		test_labels = f['species_labels'][:]
	f.close()

	# restore graph
	sess = tf.Session()
	saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
	saver.restore(sess,'./' + model_dir + model_name)

	graph = tf.get_default_graph()

	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	rep = graph.get_tensor_by_name('representationAll/Relu:0')
	preds_list = [graph.get_tensor_by_name('dense_'+str(i+1)+'/Softmax:0') for i in range(test_labels.shape[1])]
	train_step = graph.get_operation_by_name('Adam')

	# define placeholders for binary species labels
	# labels_list = [graph.get_tensor_by_name("label" + str(i) + ":0") for i in range(test_labels.shape[1])]

	labels_list = [graph.get_tensor_by_name("Placeholder:0")]
	labels_list.extend([graph.get_tensor_by_name("Placeholder_"+str(i)+":0") for \
		i in range(1,test_labels.shape[1])])

	prediction_names = []
	if write2file:
		for i in range(0,test_dat.shape[0],50):
			# run session and get output of representational layer
			feed_dict = {labels_list[j]: test_labels[i:i+50,j] for j in range(test_labels.shape[1])}
			feed_dict.update({dna: test_dat[i:i+50], dropout1: 0})
			pred_probs_list = sess.run(preds_list,feed_dict)

			# f = open(model_name + '_predProbs.txt','a')
			# writer = csv.writer(f,delimiter='\t')
			# for j in range(pred_probs.shape[0]):
			# 	writer.writerow(pred_probs[j])
			# f.close()

			# prediction_names.extend([label_names[np.argmax(pred_probs[i])] \
			# 	for i in range(pred_probs.shape[0])])
	else:
		feed_dict = {labels_list[j]: test_labels[:,j] for j in range(test_labels.shape[1])}
       	feed_dict.update({dna: test_dat, dropout1: 0})
       	pred_probs_list = sess.run(preds_list,feed_dict)

	# truelabel_names = f['species'][0:10000]
	# truelabel_names = [str(i) for i in truelabel_names]
	# truelabel_names = [label_names[np.argmax(label)] for label in test_labels]
	# prediction_names = [label_names[np.argmax(pred)] for pred in pred_probs]

	return pred_probs_list

def convertBinPreds2Flat(pred_probs_list):

	preds = np.array([1*(pred_arr[:,0] > 0.5) for pred_arr in pred_probs_list])
	
	return preds.T

def convertBinLabel2Flat(binlabel_arr):

	return np.array([binlabel_arr[i][:,0] for i in range(binlabel_arr.shape[0])])

def predPrecisionRecall(pred_probs_list,binlabel_arr):

	preds = [1*(pred_arr[:,0] > 0.5) for pred_arr in pred_probs_list]
	truePos = [preds[i]*binlabel_arr[:,i,0] for i in range(len(preds))]
	recall = [sum(truePos[i])/sum(binlabel_arr[:,i,0]) for i in range(len(preds))]
	precision = [sum(truePos[i])/sum(preds[i]) for i in range(len(preds))]

	return np.nanmean(precision), np.nanmean(recall)

def predPrecisionRecall2(pred_probs_list,binlabel_arr):

	from sklearn.metrics import precision_score, recall_score
	preds = [1*(pred_arr[:,0] > 0.5) for pred_arr in pred_probs_list]
	posPreds = [sum(preds[i]) for i in range(len(preds))]
	print(posPreds)
	precision = [precision_score(binlabel_arr[:,i,0],preds[i]) for i in range(len(preds))]
	recall = [recall_score(binlabel_arr[:,i,0],preds[i]) for i in range(len(preds))]

	print(precision)
	print(recall)

	return np.nanmean(precision), np.nanmean(recall)

def calcPrecisionRecall(cm):
	'''calculates precision and recall from an inputted confusion matrix'''

	true_pos = np.diag(cm)
	precision = true_pos / np.sum(cm, axis=0)
	recall = true_pos / np.sum(cm, axis=1)

	return precision, recall
	
### FEATURE IMPORTANCE VIA GRADIENTS ##########################################

def top_label_and_score(dnaseq,label,sess,graph,label_names):
	'''returns the predicted label and prediction scores (output layer output)
	for a given one-hot encoded DNA sequence'''

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	labels = graph.get_tensor_by_name("label:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	opt = graph.get_tensor_by_name('representation/Relu:0')
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	feed_dict = {dna: dnaseq, labels: label, dropout1: 0}
	pred_score = sess.run(preds,feed_dict)	

	maxscoreIdx = np.argmax(pred_score)
	trueIdx = np.argmax(label)

	return label_names[maxscoreIdx], label_names[trueIdx], \
		pred_score[0][maxscoreIdx]

def gradients(sess,graph,dnaseq):
	'''computes the attributions/feature importances for the prediction
	label based on the gradients at the inputted DNA sequence'''

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# tensor representing the softmax output for a given label
	preds_label = tf.reduce_mean(preds,reduction_indices=0)

	feed_dict = {dna: np.array([dnaseq]), dropout1: 0}
	pred_score = sess.run(preds,feed_dict)	

	maxscoreIdx = np.argmax(pred_score)

	grad = tf.gradients(preds_label[maxscoreIdx],dna)[0]
	grad_vals = sess.run(grad,feed_dict)

	return grad_vals*dnaseq

def scale_dnaseq(dnaseq,alpha):
	'''scales the one-hot encoded DNA sequence by an inputted scaling parameter
	with [0.25,0.25,0.25,0.25] as the baseline representation'''

	scaled_seq = [[],[],[],[]]
	for i in range(dnaseq.shape[1]):
		for j in range(4): # iterate through ATCG code of each base pair
			if dnaseq[j][i] == 1.:
				updated_val = 1.-0.75*alpha
				scaled_seq[j].append(np.array([updated_val]))
			else:
				updated_val = 0.25*alpha
				scaled_seq[j].append(np.array([updated_val]))

	scaled_seq = [np.array(i) for i in scaled_seq]
	return np.array(scaled_seq)

def integrated_gradients(sess,graph,dnaseq,labelIdx=False):
	'''computes the attributions/feature importances for the prediction
	label based on integrated gradients at the inputted DNA sequence'''

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# tensor representing the softmax output for a given label
	preds_label = tf.reduce_mean(preds,reduction_indices=0)


	if labelIdx: # identify attributions associated with given label
		scoreIdx = labelIdx
	else: # identify attributions associated with highest predicted label
		feed_dict = {dna: np.array([dnaseq]), dropout1: 0}
		pred_score = sess.run(preds,feed_dict)	
		scoreIdx = np.argmax(pred_score)

	# scale one-hot encoded DNA sequences
	scaled_dna = np.array([scale_dnaseq(dnaseq,i) for i in np.linspace(0,1,50)])
	feed_dict2 = {dna: scaled_dna,dropout1: 0}

	grad = tf.gradients(preds_label[scoreIdx],dna)[0]
	grad_vals = sess.run(grad,feed_dict2)

	return dnaseq*np.average(grad_vals,axis=0)

def integrated_gradientsBinary(sess,graph,dnaseq,num_species,labelIdx):
	'''computes the attributions/feature importances for the prediction
	label based on integrated gradients at the inputted DNA sequence'''


	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	rep = graph.get_tensor_by_name('representationAll/Relu:0')
	preds_list = [graph.get_tensor_by_name('dense_'+str(i+1)+'/Softmax:0') for \
		i in range(num_species)]
	train_step = graph.get_operation_by_name('Adam')

	# tensor representing the softmax output for a given label
	preds_label = tf.reduce_mean(preds_list[labelIdx],reduction_indices=0)

	# scale one-hot encoded DNA sequences
	scaled_dna = np.array([scale_dnaseq(dnaseq,i) for i in np.linspace(0,1,50)])
	feed_dict = {dna: scaled_dna,dropout1: 0}

	grad = tf.gradients(preds_label[0],dna)[0]
	grad_vals = sess.run(grad,feed_dict)

	return dnaseq*np.average(grad_vals,axis=0)

def scaleAttr_dnaseq(dnaseq,attrs,ptile=99):
	'''scales the one-hot encoded DNA sequence by the absolute value of the
	inputted attributions associated with the DNA sequence'''

	attrs = abs(attrs)
	attrs = np.clip(attrs/np.percentile(attrs,ptile),0,1)
	scaled_dna = dnaseq*attrs

	return scaled_dna

def plotAttr_dnaseq(scaled_dna):
	'''visualizes the attributions of an inputted scaled one-hot encoded DNA sequence'''

	# dictionary relating nucleotide color and index in one-hot encoded sequence
	colorIdx_dict = {0: 'green', 1: 'red', 2: 'blue', 3:'orange'}

	colors = []
	vals = []

	for i in range(scaled_dna.shape[1]):
		for j in range(4):
			if scaled_dna[j][i] > 0:
				colors.append(colorIdx_dict[j])
				vals.append(scaled_dna[j][i])
				break

	fig, ax = plt.subplots(2,figsize=(12, 6))

	vals_group = []
	for i in range(0,len(vals),5):
		vals_group.append(np.mean(vals[i:i+3]))

	ax[0].bar(range(1,len(vals)+1),vals,color=colors)
	A_patch = mpatches.Patch(color='green', label='A')
	T_patch = mpatches.Patch(color='red', label='T')
	C_patch = mpatches.Patch(color='blue', label='C')
	G_patch = mpatches.Patch(color='orange', label='G')

	ax[0].legend(handles=[A_patch,T_patch,C_patch,G_patch])

	ax[1].bar(np.array(range(1,len(vals)+1,5)),vals_group,width=5)
	plt.show()

def deepliftAttrs(keras_model,dnaseq):
	'''uses DeepLIFT to determine the attributions for an inputted one-hot
	encoded DNA sequence'''

	import deeplift
	from deeplift.conversion import keras_conversion as kc

	# predict label and identify index of label (in output)
	predict = keras_model.predict(np.array([dnaseq]))
	task_idx = np.argmax(predict)

	deeplift_model = kc.convert_sequential_model(keras_model, \
		nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

	find_scores_layer_idx = 0

	deeplift_contribs_func = deeplift_model.get_target_contribs_func(
		find_scores_layer_idx=find_scores_layer_idx,target_layer_idx=-2)

	scores = np.array(deeplift_contribs_func(task_idx=task_idx,
	                                         input_data_list=[dnaseq],
	                                         batch_size=10,
	                                         progress_update=1000))

	return scores

### DIMENSIONALITY REDUCTION AND CLUSTER ANALYSIS #############################

def write_projection(representation_file,output_file,method='PCA'):
	'''performs an inputted dimensionality reduction method on the output of 
	a representational layer and writes the projection to a .tsv file'''

	X = np.loadtxt(representation_file,delimiter='\t')

	if method == 'PCA':
		from sklearn.decomposition import PCA
		model = PCA(n_components=X.shape[1])
		transformed_X = model.fit_transform(X)

	elif method == 'tSNE':
		# from sklearn.manifold import TSNE
		from bhtsne import tsne
		# model = TSNE(n_components=10,perplexity=50,n_iter=400)
		# transformed_X = model.fit_transform(X) 
		transformed_X = tsne(X,dimensions=5)

	np.savetxt(output_file,transformed_X,delimiter='\t')

def GMM_analysis(data_file,h5_file,num_clusters,prob_threshold=0.4):
	'''uses a Gaussian mixture model to "cluster" the data projections from PCA
	or tSNE and identifies genes that have high probabilities of belonging to 
	more than 1 cluster

	NOTE: the inputted data_file should be the output of the representational layer
	or its projection from PCA or tSNE'''

	from sklearn.mixture import GaussianMixture

	# load h5 file containing gene name info
	f = h5py.File(h5_file,'r')

	# load representational layer output or its projection
	X = np.loadtxt(data_file,delimiter='\t')

	# fit GMM to data and predict posterior probabilities per each component/species
	gmm = GaussianMixture(n_components=num_clusters, covariance_type='full').fit(X)
	preds = gmm.predict_proba(X)

	# identify genes that are "between" clusters and the clusters they are between
	combos = list(itertools.combinations(range(num_clusters),2))
	betw_genes = {i:[] for i in combos}

	for i in range(preds.shape[0]):
		# get indices of prediction vector that have prob > threshold
		inds = [Idx for Idx,x in enumerate(preds[i]>prob_threshold) if x]
		if len(inds) > 1: # if multiple high probability clusters
			betw_genes[tuple(inds)].append(f['genes'][i])

	f.close()

	return betw_genes

def getUncertainPredictions(predProb_file,h5_file,species_list,prob_threshold=0.45):
	'''takes an input of prediction probabilities that correspond to labels
	in an accompanying HDF5 file and returns a dictionary of genes that have high
	probabilities of belonging to more than one species indexed by species name'''

	# load prediction probability data
	pred_probs = np.loadtxt(predProb_file,delimiter='\t')

	# open HDF5 file to get gene names, species labels
	f = h5py.File(h5_file,'r')
	genes = f['genes']
	labels = f['species_labels']

	# get indices of "uncertain" promoters in HDF5 file
	uncertainGeneIdx = [i for i in range(pred_probs.shape[0]) \
		if sum(pred_probs[i] >= prob_threshold) > 1]

	gene_dict = {species:[] for species in species_list}
	for i in uncertainGeneIdx:
		gene_dict[species_list[np.argmax(labels[i])]].append(str(genes[i]))

	f.close()

	return gene_dict

def getCertainPredictions(predProb_file,h5_file,species_list,prob_threshold=0.95):
	'''takes an input of prediction probabilities that correspond to labels
	in an accompanying HDF5 file and returns a dictionary of genes that have high
	probabilities of belonging to only a single species indexed by species name'''

	# load prediction probability data
	pred_probs = np.loadtxt(predProb_file,delimiter='\t')

	# open HDF5 file to get gene names, species labels
	f = h5py.File(h5_file,'r')
	genes = f['genes']
	labels = f['species_labels']

	# get indices of "uncertain" promoters in HDF5 file
	certainGeneIdx = [i for i in range(pred_probs.shape[0]) \
		if sum(pred_probs[i] >= prob_threshold) > 0]

	gene_dict = {species:[] for species in species_list}
	for i in certainGeneIdx:
		gene_dict[species_list[np.argmax(labels[i])]].append(str(genes[i]))

	f.close()

	return gene_dict

def clusterAccuracy(data_file,h5_file,num_clusters):
	'''clusters inputted projected representations and determines the accuracy 
	of the cluster assignments with respect to the actual labels (i.e. species) 
	given to the genes'''

	from sklearn.mixture import GaussianMixture

	# get species labels and convert to list of indices corresponding to label
	f = h5py.File(h5_file,'r')
	labels = f['species_labels'][:]
	labels = np.argmax(labels,axis=1)

	# load representational layer output or its projection
	X = np.loadtxt(data_file,delimiter='\t')

	# fit GMM to data and predict posterior probabilities per each component/species
	gmm = GaussianMixture(n_components=num_clusters, covariance_type='full').fit(X)
	pred_probs = gmm.predict_proba(X)
	preds = np.argmax(pred_probs,axis=1)

	return np.mean(labels==preds)

def getCertainSeq(certainGenes_file,uncertainGenes_file,promoterSeq_file,promoter_length,outfile):
	'''retrieves a list of genes w/ "certain" predictions and genes w/ "uncertain"
	predictions as well as their promoter sequences from an inputted sequence
	file; outputs their sequences to an HDF5 file with one-hot encoded labels 
	concerning their prediction un/certainty'''

	with open(certainGenes_file,'r') as f:
		reader = csv.reader(f,delimiter='\t')
		certainGenes = [line[0] for line in reader]

	with open(uncertainGenes_file,'r') as f:
		reader = csv.reader(f,delimiter='\t')
		uncertainGenes = [line[0] for line in reader]	

	seq_dict = read_fasta_seq(promoterSeq_file,certainGenes + uncertainGenes)
	gene_windows = getrandWindow(seq_dict,promoter_length)
	gene_list = certainGenes + uncertainGenes
	num_windows = [len(certainGenes),len(uncertainGenes)]
	
	labels = getLabels(num_windows)

	print('writing...')
	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File(outfile,'w')
	f.create_dataset('dnaseq',data=gene_windows,dtype='f',compression='gzip')
	f.create_dataset('labels',data=labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=gene_list,dtype=dt,compression='gzip')
	f.close()

# model_name = 'all10binBinary_model'
# model_dir = ''
# testdata_file = 'data/h5datasets/all10bin/all.h5'
# get_representationsBinary(model_name,model_dir,testdata_file,True)

# model_dir = ''
# model_name = 'all10_dense_model'
# testdata_file = 'data/h5datasets/all10/all.h5'
# get_representations(model_name,model_dir,testdata_file,write2file=True)

# model_name = 'all10binBinary_model'
# model_dir = ''
# testdata_file = 'data/h5datasets/all10bin/validation_even.h5'
# a = getPredictionsBinary(model_name,model_dir,testdata_file,limit=1000)

# clusterAccuracy('results/Mouse_Human/Mouse_Human_dense_allgenes_repPCA.txt','data/h5datasets/Mouse_Human/all.h5',2)

# species_list = ['sCer','cEleg','Mouse','Human','sPom','Zebrafish','dMelan','Chicken','aThal','Lizard']
# write_metadata(species_list,'all10','all.h5')

representation_file = 'results/all10bin/all10bin_MouseHumanLizardAllGenes_filteredrep.txt'
output_file = 'all10bin_MouseHumanLizardAllGenes_filteredrep_tSNE.txt'
write_projection(representation_file,output_file,method='tSNE')

# model_name = 'Mouse_Human_dense_model' # model is densely connected network
# model_dir = 'results/Mouse_Human/'
# testdata_file = 'data/h5datasets/Mouse_Human/all.h5'
# label_names = ['Mouse','Human']

# # getPredictions(model_name,model_dir,testdata_file,label_names,write2file=True)

# species_list = label_names
# predProb_file = 'Mouse_Human_dense_model_predProbs.txt'
# h5_file = testdata_file
# blah = getUncertainPredictions(predProb_file,h5_file,species_list,prob_threshold=0.4)
# # get representations for all genes and write to file


# write_metadata(['Mouse','Human'],'Mouse_Human','all.h5')

# representation_file = 'results/all10bin/all10binBinary_model_repAllGenes.txt'
# metadata_file = 'results/all10/all10_metadataAllGenes.tsv'
# index_set = [2,3,9]
# output_file = 'all10bin_MouseHumanLizardAllGenes'
# filterLabels(representation_file,metadata_file,index_set,output_file)

# sess = tf.Session()
# saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
# saver.restore(sess,'./' + model_dir + model_name)

# graph = tf.get_default_graph()
	

# keras_model_weights = 'sCer_dHansmodel.h5'
# keras_model_json = 'sCer_dHansmodel.json'

# keras_model = kc.load_keras_model(weights=keras_model_weights,
#                                   json=keras_model_json)

# f = h5py.File('data/h5datasets/sCer_dHans/validation.h5','r')

# a = deepliftAttrs(keras_model,f['dnaseq'][0])

# f.close()

# representation_file = 'results/saved_models/sCer_cEleg_Mouse_Human/sCer_cEleg_Mouse_Human_val_rep.txt'
# metadata_file = 'results/saved_models/sCer_cEleg_Mouse_Human/sCer_cEleg_Mouse_Human_val_metadata.tsv'
# label_set = ['2','3']
# output_file = 'all4_Mouse_Human.tsv'

# filterLabels(representation_file,metadata_file,label_set,output_file)

# model_name = 'sCer_cEleg_Mouse_Human_model'
# model_dir = 'results/saved_models/sCer_cEleg_Mouse_Human/'
# testdata_file = 'data/h5datasets/sCer_cEleg_Mouse_Human/validation.h5'

# write_metadata(['sCer','cEleg','Mouse','Human','sPom','Zebrafish','dMelan','sBoul'],'all8','validation.h5')
# write_metadata(['sCer','cEleg','Mouse','Human'],'all4_unmasked','validation.h5')
# write_metadataGO(['stress','cell_cycle','chromosome_org','cell_loc','kinase','pyrophos','multiple'],'new_sCer_6GO')
# write_metadataGO(['stress','cc','chr_org','multiple'],'new_sCer_stress+cc+chrorg')


# data_file = 'results/saved_models/sCer_cEleg_Mouse_Human/blah.txt'
# h5_file = 'data/h5datasets/sCer_cEleg_Mouse_Human/validation.h5'
# num_clusters = 4

# a = GMM_analysis(data_file,h5_file,num_clusters,prob_threshold=0.4)
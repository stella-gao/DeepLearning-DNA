import csv
import numpy as np
import h5py
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf

### METADATA ##################################################################

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
		metadata_Idxs = []
		metadata_labels = []
		# for j in range(f['labels'].shape[1]):
		# 	if f['labels'][i][j] == 1:
		# 		metadata_Idxs.append(str(j))
		# 		metadata_labels.append(label_set[j])

		if sum(f['labels'][i]) >= 2:
			labelIdx = len(label_set)-1
		else:
			labelIdx = list(f['labels'][i]).index(max(f['labels'][i]))
		writer.writerow([labelIdx,label_set[labelIdx]])
		# writer.writerow(['+'.join(metadata_Idxs),'+'.join(metadata_labels)])

	f.close()
	g.close()


def filterLabels(representation_file,metadata_file,index_set,output_file):
	'''takes in a .txt file of representational layer outputs along with the
	corresponding raw data file (in H5 format) and filters out the 
	representational layer outputs that are associated with an inputted label'''

	rep_dat = np.loadtxt(representation_file,delimiter='\t')

	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')

	g = open(output_file,'w')
	writer = csv.writer(g,delimiter='\t')
	
	i = 0
	index_list = []
	writer.writerow(reader.next()) # skip metadata file header
	for line in reader:
		if line[0] in index_set:
			writer.writerow(line)
			index_list.append(i)
		i += 1

	np.savetxt(output_file + '_filteredrep.txt',rep_dat[np.array(index_list)], \
		delimiter='\t')

	f.close()
	g.close()

### CNN LAYER OUTPUTS #########################################################

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
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# load test data
	f = h5py.File(testdata_file,'r')
	test_dat = f['dnaseq'][:]
	test_labels = f['labels'][:]
	f.close()

	# internal accuracy
	correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# run session and get output of representational layer
	feed_dict = {dna: test_dat, labels: test_labels, dropout1: 0}
	rep_output,acc = sess.run([opt,accuracy],feed_dict)

	print(acc)

	if write2file:
		np.savetxt(model_name + '_rep.txt',rep_output,delimiter='\t')

	else:
		return rep_output

def getPredictions(model_name,model_dir,testdata_file,label_names):
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
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# load test data
	f = h5py.File(testdata_file,'r')
	test_dat = f['dnaseq'][:]
	test_labels = f['labels'][:]
	f.close()

	# run session and get output of representational layer
	feed_dict = {dna: test_dat, labels: test_labels, dropout1: 0}
	predictions = sess.run(preds,feed_dict)

	truelabel_names = [label_names[np.argmax(label)] for label in test_labels]
	prediction_names = [label_names[np.argmax(pred)] for pred in predictions]

	return predictions,confusion_matrix(truelabel_names,prediction_names)

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

def integrated_gradients(sess,graph,dnaseq):
	'''computes the attributions/feature importances for the prediction
	label based on integrated gradients at the inputted DNA sequence'''

	# identify relevant placeholders and operations
	dna = graph.get_tensor_by_name("dna:0")
	dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
	preds = graph.get_tensor_by_name('dense_1/Softmax:0')

	# tensor representing the softmax output for a given label
	preds_label = tf.reduce_mean(preds,reduction_indices=0)

	# identify highest predicted label
	feed_dict = {dna: np.array([dnaseq]), dropout1: 0}
	pred_score = sess.run(preds,feed_dict)	
	maxscoreIdx = np.argmax(pred_score)

	# scale one-hot encoded DNA sequences
	scaled_dna = np.array([scale_dnaseq(dnaseq,i) for i in np.linspace(0,1,50)])
	feed_dict2 = {dna: scaled_dna,dropout1: 0}

	grad = tf.gradients(preds_label[maxscoreIdx],dna)[0]
	grad_vals = sess.run(grad,feed_dict2)

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
	colorIdx_dict = {0: 'red', 1: 'blue', 2: 'green', 3:'yellow'}

	colors = []
	vals = []

	for i in range(scaled_dna.shape[1]):
		for j in range(4):
			if scaled_dna[j][i] > 0:
				colors.append(colorIdx_dict[j])
				vals.append(scaled_dna[j][i])
				break

	fig, ax = plt.subplots(2) #, figsize=(6, 12))

	vals_group = []
	for i in range(0,len(vals),10):
		# vals_group.append(np.mean(vals[i:i+10]))
		vals_group.append(i)

	print(vals_group)

	ax[0].bar(range(1,len(vals)+1),vals,color=colors)
	print(np.array(range(1,len(vals)+1,10))+5)
	ax[1].bar(np.array(range(1,len(vals)+1,10))+5,vals_group)
	plt.show()

### DIMENSIONALITY REDUCTION ##################################################

def write_projection(representation_file,output_file,method='PCA'):
	'''performs an inputted dimensionality reduction method on the output of 
	a representational layer and writes the projection to a .tsv file'''

	X = np.loadtxt(representation_file,delimiter='\t')

	if method == 'PCA':
		from sklearn.decomposition import PCA
		model = PCA(n_components=2)
		transformed_X = model.fit_transform(X)

	elif method == 'tSNE':
		from sklearn.manifold import TSNE
		model = TSNE(n_components=2)
		transformed_X = model.fit_transform(X) 

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
	print(gmm.means_)
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


representation_file = 'results/saved_models/sCer_cEleg_Mouse_Human/sCer_cEleg_Mouse_Human_val_rep.txt'
metadata_file = 'results/saved_models/sCer_cEleg_Mouse_Human/sCer_cEleg_Mouse_Human_val_metadata.tsv'
label_set = ['2','3']
output_file = 'all4_Mouse_Human.tsv'

# filterLabels(representation_file,metadata_file,label_set,output_file)

model_name = 'sCer_cEleg_Mouse_Human_model'
model_dir = 'results/saved_models/sCer_cEleg_Mouse_Human/'
testdata_file = 'data/h5datasets/sCer_cEleg_Mouse_Human/validation.h5'

label_names = ['sCer','cEleg','Mouse','Human']
# a = getPredictions(model_name,model_dir,testdata_file,label_names)
# print a[1]
# a = get_representations(model_name,model_dir,testdata_file)

# file_name = 'all.h5'
# write_metadata(['sCer','sPom'],'sCer_sPom',file_name)
# d = np.loadtxt('sCer_sPom_rep.txt',delimiter='\t')


# write_metadata(['sCer','sBoul','sArb','sEub'],'new2_Sac4','validation.h5')
# write_metadataGO(['stress','cell_cycle','chromosome_org','cell_loc','kinase','pyrophos','multiple'],'sCer_stress+cell_cycle+chromosome_org+cell_loc+kinase+pyrophos')

# data_file = 'results/saved_models/sCer_cEleg_Mouse_Human/blah.txt'
# h5_file = 'data/h5datasets/sCer_cEleg_Mouse_Human/validation.h5'
# num_clusters = 4

# a = GMM_analysis(data_file,h5_file,num_clusters,prob_threshold=0.4)

# restore graph
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
saver.restore(sess,'./' + model_dir + model_name)

f = h5py.File(testdata_file,'r')
dnaseq = f['dnaseq'][0]
lab = f['labels'][0]

graph = tf.get_default_graph()

# identify relevant placeholders and operations
dna = graph.get_tensor_by_name("dna:0")
labels = graph.get_tensor_by_name("label:0")
dropout1 = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
opt = graph.get_tensor_by_name('representation/Relu:0')
preds = graph.get_tensor_by_name('dense_1/Softmax:0')

feed_dict = {dna: dnaseq, labels: lab, dropout1: 0}

grads_attrs = gradients(sess,graph,dnaseq)
scaled_dna = scaleAttr_dnaseq(dnaseq,grads_attrs,ptile=99)
plotAttr_dnaseq(scaled_dna[0])


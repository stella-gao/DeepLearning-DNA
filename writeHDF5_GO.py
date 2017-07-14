import h5py
import numpy as np
from model import *

upstream_length = 1000
promoter_length = 500
window_step = 20

def combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step,sampling_method=1,random_sample=False,balance=False):

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
	    sp in species]
	ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]

	# read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = {species_list[i]:read_fasta_seq(seq_files[i]) for i \
	    in range(len(species_list))}

	# read in GO terms and corresponding genes for each species
	GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],species_list) \
	    for i in range(len(ontology_files))}			

	if balance: # ensure equal number of genes per GO terms per species
		minGOterms = min([min([len(GO_dicts[GOterm][species_name]) for species_name \
			in GO_dicts[GOterm]]) for GOterm in GO_dicts])
		GO_dicts = {GOterm: {species_name: list(np.random.choice(GO_dicts[GOterm][species_name],\
			size=minGOterms,replace=False))} for GOterm in GO_dicts}

	# creates a dictionary of ontologies indexed by species+gene names
	geneterm_dict = geneGO_dict(GO_dicts)

	if sampling_method == 1: # construct matrix of all windows and stratify
		# create windows and labels for windows for all genes/regions
		mat,GO_labels,gene_list = multilabelWindows(seq_dicts,geneterm_dict,ontology_terms,\
		                    promoter_length,window_step,random_sample)

		# stratify data into training and validation data
		allIdx = range(len(mat))
		np.random.shuffle(allIdx)
		trainIdx = allIdx[0:int(0.9*len(mat))]
		valIdx = allIdx[int(0.9*len(mat)):]

		train_dat = np.array([mat[i] for i in trainIdx])
		train_labels = np.array([GO_labels[i] for i in trainIdx])
		train_genelist = [gene_list[i] for i in trainIdx]

		validation_dat = np.array([mat[i] for i in valIdx])
		validation_labels = np.array([GO_labels[i] for i in valIdx])
		validation_genelist = [gene_list[i] for i in valIdx]

	elif sampling_method == 2: # stratify based on species+gene terms

		# isolate genes to be included in training and validation datasets
		train_geneterms = list(np.random.choice(geneterm_dict.keys(),\
			int(len(geneterm_dict)*0.9),replace=False))
		validation_geneterms = [geneterm for geneterm in geneterm_dict.keys() \
			if geneterm not in train_geneterms]

		train_geneterm_dict = {key:geneterm_dict[key] for key in geneterm_dict.keys() \
			if key in train_geneterms}
		val_geneterm_dict = {key:geneterm_dict[key] for key in geneterm_dict.keys() \
			if key in validation_geneterms}

		# get windows, labels, species+gene names for training, validation data
		train_dat,train_labels,train_genelist = \
			multilabelWindows(seq_dicts,train_geneterm_dict,\
			ontology_terms,promoter_length,window_step,random_sample,20)

		validation_dat,validation_labels,validation_genelist = \
			multilabelWindows(seq_dicts,val_geneterm_dict,\
			ontology_terms,promoter_length,window_step,random_sample,20)

		if random_sample:
			print('shuffling to create random sequences...')

			# random training data
			random_trainsize = train_dat.shape[0]/len(ontology_terms)
			random_train = train_dat[np.random.choice(range(train_dat.shape[0]),\
				random_trainsize)]
			random_train = np.array([shuffle_onehotseq(random_train[i]) for \
				i in range(random_train.shape[0])])
			train_dat = np.concatenate([train_dat,random_train])

			# # random validation data
			# random_valsize = validation_dat.shape[0]/len(ontology_terms)
			# random_val = validation_dat[np.random.choice(range(validation_dat.shape[0]),\
			# 	random_valsize)]
			# random_val = np.array([shuffle_onehotseq(random_val[i]) for \
			# 	i in range(random_val.shape[0])])
			# validation_dat = np.concatenate([validation_dat,random_val])

			# random training, validation labels
			random_label = np.zeros(len(ontology_terms)+1,dtype=np.float32)
			random_label[len(ontology_terms)] = 1.
			train_labels = np.concatenate([train_labels,\
				np.array([random_label for i in range(random_trainsize)])])
			# validation_labels = np.concatenate([validation_labels,\
			# 	np.array([random_label for i in range(random_valsize)])])	

			# random training, validation genelist
			train_genelist.extend(['random' for i in range(random_trainsize)])
			# validation_genelist.extend(['random' for i in range(random_valsize)])		

		# shuffle training data (to randomize batch-species pairs)
		train_shuffleIdx = range(train_dat.shape[0])
		np.random.shuffle(train_shuffleIdx)
		train_dat = train_dat[np.array(train_shuffleIdx)]
		train_labels = [train_labels[i] for i in train_shuffleIdx]
		train_genelist = [train_genelist[i] for i in train_shuffleIdx]

		# shuffle validation data (to randomize species+gene order)
		val_shuffleIdx = range(validation_dat.shape[0])
		np.random.shuffle(val_shuffleIdx)
		validation_dat = validation_dat[np.array(val_shuffleIdx)]
		validation_labels = [validation_labels[i] for i in val_shuffleIdx]
		validation_genelist = [validation_genelist[i] for i in val_shuffleIdx]

	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('GO_labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genelist,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('GO_labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genelist,dtype=dt,compression='gzip')
	f.close()

def convertLabels(labels):
	'''converts an array of multi-label vectors into array containing arrays
	of binary labels'''

	converted_labels = []
	for i in range(labels.shape[0]):	
		bin_labels = []
		for j in range(labels.shape[1]):
			if labels[i][j] == 1:
				bin_labels.append(np.array([1.,0.]))
			else:
				bin_labels.append(np.array([0.,1.]))
		converted_labels.append(np.array(bin_labels))

	return np.array(converted_labels)

# species = ['sCer'] #,'cEleg','Mouse','Human']
# species_list = ['cerevisiae'] #,'elegans','musculus','sapiens']

# ontology_terms = ['stressP','cell_cycleP'] #,'chromosome_orgP'] #,'cell_loc','kinase','pyrophos']
# random_sample = True

random_sample = False
# ontology_terms = ['translation','cell_cycleP']
ontology_terms = ['skeletal','multicell_dev']
species = ['Human']
species_list = ['sapien']

combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step,2,random_sample,balance=False)



# ontology_terms = ['stress','cell_cycle','chromosome_org','cell_loc','kinase','pyrophos']

# combine_GOterms(species,species_list,ontology_terms,upstream_length,\
# 		promoter_length,window_step,2)

def rewriteHDF5_GO(h5_file,dir_name):

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('GO_label',data=f['labels'][:],dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

def rewriteHDF5_GO2(h5_file,dir_name): # multi-label binary encoding

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	GO_labels = convertLabels(f['GO_labels'][:])

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('GO_labels',data=GO_labels,dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

# dir_name = 'data/h5datasets_GO/Human_skel_multidev/'
# rewriteHDF5_GO2('train.h5',dir_name)
# rewriteHDF5_GO2('validation.h5',dir_name)

# dir_name = 'data/h5datasets_GO/new_sCer_6GO/'
# rewriteHDF5_GO('train.h5',dir_name)
# rewriteHDF5_GO('validation.h5',dir_name)

# dir_name = 'data/h5datasets_GO/new_sCer_stress+cc+chrorg/'
# rewriteHDF5_GO('train.h5',dir_name)
# rewriteHDF5_GO('validation.h5',dir_name)

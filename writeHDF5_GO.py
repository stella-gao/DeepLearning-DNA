import h5py
import numpy as np
from model import *

species = ['sCer','cEleg','Mouse','Human']
species_list = ['cerevisiae','elegans','musculus','sapiens']
ontology_terms = ['stress','cell_cycle','chromosome_org','cell_loc','kinase','pyrophos']

upstream_length = 1000
promoter_length = 500
window_step = 20

def combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step,sampling_method=1):

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
	    sp in species]
	ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]

	# read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = {species_list[i]:read_fasta_seq(seq_files[i]) for i \
	    in range(len(species_list))}

	# read in GO terms and corresponding genes for each species
	GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],species_list) \
	    for i in range(len(ontology_files))}

	# creates a dictionary of ontologies indexed by species+gene names
	geneterm_dict = geneGO_dict(GO_dicts)

	if sampling_method == 1: # construct matrix of all windows and stratify
		# create windows and labels for windows for all genes/regions
		mat,GO_labels,gene_list = multilabelWindows(seq_dicts,geneterm_dict,ontology_terms,\
		                    promoter_length,window_step)

		# stratify data into training and validation data
		allIdx = range(len(mat))
		np.random.shuffle(allIdx)
		trainIdx = allIdx[0:int(0.8*len(mat))]
		valIdx = allIdx[int(0.8*len(mat)):]

		train_dat = np.array([mat[i] for i in trainIdx])
		train_labels = np.array([GO_labels[i] for i in trainIdx])
		train_genelist = [gene_list[i] for i in trainIdx]

		validation_dat = np.array([mat[i] for i in valIdx])
		validation_labels = np.array([GO_labels[i] for i in valIdx])
		validation_genelist = [gene_list[i] for i in valIdx]

	elif sampling_method == 2: # stratify based on species+gene terms

		# isolate genes to be included in training and validation datasets
		train_geneterms = list(np.random.choice(geneterm_dict.keys(),\
			int(len(geneterm_dict)*0.8),replace=False))
		validation_geneterms = [geneterm for geneterm in geneterm_dict.keys() \
			if geneterm not in train_geneterms]

		train_geneterm_dict = {key:geneterm_dict[key] for key in geneterm_dict.keys() \
			if key in train_geneterms}
		val_geneterm_dict = {key:geneterm_dict[key] for key in geneterm_dict.keys() \
			if key in validation_geneterms}

		# get windows, labels, species+gene names for training, validation data
		train_dat,train_labels,train_genelist = \
			multilabelWindows(seq_dicts,train_geneterm_dict,\
			ontology_terms,promoter_length,window_step)

		validation_dat,validation_labels,validation_genelist = \
			multilabelWindows(seq_dicts,val_geneterm_dict,\
			ontology_terms,promoter_length,window_step)

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

ontology_terms = ['stress','cell_cycle','chromosome_org'] #,'cell_loc','kinase','pyrophos']

combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step,2)

ontology_terms = ['stress','cell_cycle','chromosome_org','cell_loc','kinase','pyrophos']

combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step,2)

def rewriteHDF5_GO(h5_file,dir_name):

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('GO_label',data=f['labels'][:],dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

# dir_name = 'data/h5datasets_GO/new_sCer_6GO/'
# rewriteHDF5_GO('train.h5',dir_name)
# rewriteHDF5_GO('validation.h5',dir_name)

# dir_name = 'data/h5datasets_GO/new_sCer_stress+cc+chrorg/'
# rewriteHDF5_GO('train.h5',dir_name)
# rewriteHDF5_GO('validation.h5',dir_name)

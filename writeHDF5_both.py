import h5py
import numpy as np
from model import *

species = ['sCer','cEleg','Mouse','Human']
species_names = ['cerevisiae','elegans','musculus','sapiens']
ontology_terms = ['stress','cell_cycle','chromosome_org'] #,'cell_loc','kinase','pyrophos']

upstream_length = 1000
promoter_length = 500
window_step = 20

def species2labels(species_names,species_list):
	'''converts an inputted list of species to one-hot encoded labels
	NOTE: species_list is the list of species to be labeled in order of the labels
	to be generated'''

	speciesIdx_dict = {species: species_names.index(species) for species \
		in species_names}

	labels = []
	for species in species_list:
		lab = np.zeros(len(species_names),dtype=np.float32)
		lab[speciesIdx_dict[species]] = 1.
		labels.append(lab)

	return (np.array(labels))


def combine_speciesGO(species,species_names,ontology_terms,upstream_length,\
		promoter_length,window_step):

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
	    sp in species]
	ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]

	# read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = {species_names[i]:read_fasta_seq(seq_files[i]) for i \
	    in range(len(species_names))}

	# read in GO terms and corresponding genes for each species
	GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],species_names) \
	    for i in range(len(ontology_files))}

	# creates a dictionary of ontologies indexed by species+gene names
	geneterm_dict = geneGO_dict(GO_dicts)

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

	# separate species from species+gene lists in training, validation data
	train_specieslist = [n.split('~')[0] for n in train_genelist]
	validation_specieslist = [n.split('~')[0] for n in validation_genelist]

	# create one-hot encoded species labels for training, validation data
	train_specieslabels = species2labels(species_names,train_species_list)
	validation_specieslabels = species2labels(species_names,validation_species_list)

	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('GO_labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('species_labels',data=train_specieslabels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genelist,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('GO_labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('species_labels',data=validation_specieslabels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genelist,dtype=dt,compression='gzip')
	f.close()


combine_speciesGO(species,species_names,ontology_terms,upstream_length,\
		promoter_length,window_step)

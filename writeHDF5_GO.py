import h5py
import numpy as np
from model import *

species = ['sCer'] #,'cEleg']
species_list = ['cerevisiae'] #,'elegans']
ontology_terms = ['stress','cell_cycle','chromosome_org']

upstream_length = 1000
promoter_length = 500
window_step = 20

def combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step):

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

	# create windows and labels for windows for all genes/regions
	(mat,GO_labels) = multilabelWindows(seq_dicts,geneterm_dict,ontology_terms,\
	                    promoter_length,window_step)

	# stratify data into training and validation data
	allIdx = range(len(mat))
	np.random.shuffle(allIdx)
	trainIdx = allIdx[0:int(0.8*len(mat))]
	valIdx = allIdx[int(0.8*len(mat)):]
	train_dat = np.array([mat[i] for i in trainIdx])
	train_labels = np.array([GO_labels[i] for i in trainIdx])
	validation_dat = np.array([mat[i] for i in valIdx])
	validation_labels = np.array([GO_labels[i] for i in valIdx])

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=train_labels,dtype='f',compression='gzip')
	f.close()

	f = h5py.File(str(species[0]) + '+'.join(ontology_terms) + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=validation_labels,dtype='f',compression='gzip')
	f.close()

combine_GOterms(species,species_list,ontology_terms,upstream_length,\
		promoter_length,window_step)
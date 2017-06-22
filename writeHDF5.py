import h5py
import numpy as np
from model import *

species = 'cEleg'
analysis = 'species'

upstream_length = 1000
promoter_length = 500
window_step = 20

def write_species(species,analysis,window_step,upstream_length,promoter_length):
	'''writes species-specific DNA sequences to file in H5 format'''

	if analysis == 'species':

		f = h5py.File(species + '.h5','w')

		seq_file = 'data/my_promoters/' + species + str(upstream_length) + '.fa.txt'

		seq_dict = read_fasta_seq(seq_file)

		windows = getallWindows(seq_dict,promoter_length,window_step)

		f.create_dataset('sequences',data=windows,dtype='f',compression='gzip')

		f.close()

def combine_species(species_list,upstream_length,promoter_length,window_step):
	'''combines sequence data from all species into one large numpy array,
	creates species labels for all sequences in the array, and writes the 
	data to H5 datasets'''

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
    sp in species_list]

    # read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = [read_fasta_seq(f) for f in seq_files]

	# set size of examples for each species to be equal to each other
	min_datsize = min([len(seq_dict) for seq_dict in seq_dicts])
	subsetIdx = [np.random.choice(range(len(sd)),size=min_datsize,replace=False) \
	    for sd in seq_dicts]
	subset_regs = [[seq_dicts[i].keys()[j] for j in subsetIdx[i]] for i \
	    in range(len(seq_dicts))]
	new_seq_dicts = [{reg:seq_dicts[i][reg] for reg in subset_regs[i]} for i \
	    in range(len(seq_dicts))]

	# use sliding windows to get windows of DNA sequences for each region
	print('creating windows...')
	gene_windows = [getallWindows(seq_dict,promoter_length,window_step) \
		for seq_dict in new_seq_dicts]
	windows = [wind for wind,gene in gene_windows]
	genes = [gene for wind,gene in gene_windows]
	gene_list = [gene for gene_set in genes for gene in gene_set]

	num_windows = [len(windows_set) for windows_set in windows] # windows/species
	print('concatenating...')
	# arrange sequence data into single array
	mat = np.concatenate(windows)

	# create species labels
	sp_labels = getLabels(num_windows)

	# stratify data into training and validation data
	allIdx = range(len(mat))
	np.random.shuffle(allIdx)
	trainIdx = allIdx[0:int(0.8*len(mat))]
	valIdx = allIdx[int(0.8*len(mat)):]

	train_genes = [gene_list[i] for i in trainIdx]
	train_dat = np.array([mat[i] for i in trainIdx])
	train_labels = np.array([sp_labels[i] for i in trainIdx])

	validation_genes = [gene_list[i] for i in valIdx]
	validation_dat = np.array([mat[i] for i in valIdx])
	validation_labels = np.array([sp_labels[i] for i in valIdx])

	print('writing...')
	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File('+'.join(species_list) + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genes,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File('+'.join(species_list) + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genes,dtype=dt,compression='gzip')
	f.close()

upstream_length = 1000
promoter_length = 500
window_step = 20

# species_list = ['sCer','dHans']

# combine_species(species_list,upstream_length,promoter_length,window_step)

# species_list = ['sCer','cEleg']

# combine_species(species_list,upstream_length,promoter_length,window_step)

# species_list = ['sCer','aNid']

# combine_species(species_list,upstream_length,promoter_length,window_step)

# species_list = ['cEleg','sPom']

# combine_species(species_list,upstream_length,promoter_length,window_step)

species_list = ['sCer','cEleg','Mouse','Human']

combine_species(species_list,upstream_length,promoter_length,window_step)

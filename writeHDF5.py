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

def combine_species(species_list,upstream_length,promoter_length,window_step,sampling_method=1):
	'''combines sequence data from all species into one large numpy array,
	creates species labels for all sequences in the array, and writes the 
	data to H5 datasets'''

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
		sp in species_list]

	print('reading in sequences...')
    # read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = [read_fasta_seq(f) for f in seq_files]

	if sampling_method == 1: # select all windows from randomly selected genes
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

	else: # select randomly selected windows across all genes
		print('creating windows...')
		# get all possible windows for genes across all species
		gene_windows = [getallWindows(seq_dict,promoter_length,window_step) \
			for seq_dict in seq_dicts]

		# randomly choose windows, genes for all windows, genes in each species
		# (ensure number of windows represented is the same for each species)
		min_datsize = min([len(gene_window[1]) for gene_window in gene_windows])
		randIdx_list = [np.random.choice(range(len(gene_window[1])), \
				size=min_datsize,replace=False) for gene_window in gene_windows]
		windows = [gene_windows[i][0][randIdx_list[i]] for i in \
			range(len(gene_windows))]
		genes = [[gene_windows[i][1][j] for j in randIdx_list[i]] for i in \
			range(len(gene_windows))]

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

def combine_species_allgenes(species_list,upstream_length,promoter_length):
	'''combines sequence data from all species into one large numpy array,
	creates species labels for all sequences in the array, and writes the 
	data to H5 datasets - ensures all genes represented by one randomly selected
	window

	NOTE: chooses ONE randomly selected window for each gene'''

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
    sp in species_list]

    # read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = [read_fasta_seq(f) for f in seq_files]

	print('creating windows...')
	# get a one-hot encoded sequence for each gene (randomly selected upstream)
	gene_windows = [getrandWindow(seq_dict,promoter_length) for seq_dict in seq_dicts]

	genes = [[gene for gene in sorted(seq_dict.keys())] for seq_dict in seq_dicts]
	gene_list = [gene for gene_set in genes for gene in gene_set]
	num_windows = [len(seq_dict) for seq_dict in seq_dicts]

	print('concatenating...')
	# arrange sequence data into single array
	mat = np.concatenate(gene_windows)

	# create species labels
	sp_labels = getLabels(num_windows)

	print('writing...')
	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File('+'.join(species_list) + '.all.h5','w')
	f.create_dataset('dnaseq',data=mat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=sp_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=gene_list,dtype=dt,compression='gzip')
	f.close()

upstream_length = 1000
promoter_length = 500
window_step = 20

species_list = ['sCer','cEleg','Mouse','Human']

seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
sp in species_list]

# combine_species_allgenes(species_list,upstream_length,promoter_length)
combine_species(species_list,upstream_length,promoter_length,window_step,2)

# seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
# sp in species_list]

# # read in promoter sequences and convert to one-hot encoding sequences
# seq_dicts = [read_fasta_seq(f) for f in seq_files]
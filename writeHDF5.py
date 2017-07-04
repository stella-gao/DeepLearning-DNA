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

	if sampling_method == 1 or sampling_method == 2: 
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

		elif sampling_method == 2: # select randomly selected windows across all genes
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

	else:
		print('creating windows...')
		gene_windows = [getallWindows(seq_dict,promoter_length,window_step,\
			concat=False) for seq_dict in seq_dicts]
		windows = [wind for wind,gene in gene_windows]
		gene_list = [[gene for gene in sorted(seq_dict)] for seq_dict in seq_dicts]

		min_datsize = min([len(gene_window[1]) for gene_window in gene_windows])
		# determine number of windows per gene to select for each species
		num_windows_per_gene = [int(float(min_datsize)/len(seq_dict.keys())) \
			for seq_dict in seq_dicts]

		# get indices for training and validation datasets
		allIdx = [range(window_set.shape[0]) for window_set in windows]
		[np.random.shuffle(inds) for inds in allIdx]
		trainIdx = [np.array(inds[0:int(0.8*len(inds))]) for inds in allIdx]
		valIdx = [np.array(inds[int(0.8*len(inds)):]) for inds in allIdx]

		print('isolating training/validation windows...')
		# randomly select windows for each gene (based on number of windows determined
		# in num_windows_per_gene above)
		train_windows = [np.concatenate(windows[i][trainIdx[i],:]\
			[:,np.random.choice(range(windows[i].shape[1]),num_windows_per_gene[i],\
			replace=False)]) for i in range(len(windows))]
		validation_windows = [np.concatenate(windows[i][valIdx[i],:]\
			[:,np.random.choice(range(windows[i].shape[1]),num_windows_per_gene[i],\
			replace=False)]) for i in range(len(windows))]

		print('creating labels...')
		# create species labels for training and validation data
		num_windows_per_species_train = [window_set.shape[0] for \
			window_set in train_windows]
		num_windows_per_species_val = [window_set.shape[0] for \
			window_set in validation_windows]

		train_labels = getLabels(num_windows_per_species_train)
		validation_labels = getLabels(num_windows_per_species_val)

		# generate lists of species names
		train_species_list = [species_list[i] for i in \
			range(len(num_windows_per_species_train)) for j in \
			range(num_windows_per_species_train[i])]

		val_species_list = [species_list[i] for i in \
			range(len(num_windows_per_species_val)) for j in \
			range(num_windows_per_species_val[i])]

		# get gene labels for training and validation data
		train_genes = []
		validation_genes = []
		for i in range(len(gene_list)):
			train_geneset = [[gene_list[i][j]]*num_windows_per_gene[i] for j in trainIdx[i]]
			train_genes.extend([gene for gene_rep in train_geneset for gene in gene_rep])

			val_geneset = [[gene_list[i][j]]*num_windows_per_gene[i] for j in valIdx[i]]
			validation_genes.extend([gene for gene_rep in val_geneset for gene in gene_rep])

		# aggregate training and validation data into large matrix
		train_dat = np.concatenate(train_windows)
		validation_dat = np.concatenate(validation_windows)

		# shuffle training data (to randomize batch-species pairs)
		train_shuffleIdx = range(train_dat.shape[0])
		np.random.shuffle(train_shuffleIdx)
		train_dat = train_dat[np.array(train_shuffleIdx)]
		train_labels = [train_labels[i] for i in train_shuffleIdx]
		train_genes = [train_genes[i] for i in train_shuffleIdx]
		train_species_list = [train_species_list[i] for i in train_shuffleIdx]

		# shuffle validation data (to randomize order of genes/species)
		val_shuffleIdx = range(validation_dat.shape[0])
		np.random.shuffle(val_shuffleIdx)
		validation_dat = validation_dat[np.array(val_shuffleIdx)]
		validation_labels = [validation_labels[i] for i in val_shuffleIdx]
		validation_genes = [validation_genes[i] for i in val_shuffleIdx]
		val_species_list = [val_species_list[i] for i in val_shuffleIdx]

	print('writing...')
	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File('+'.join(species_list) + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('species_labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=train_species_list,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File('+'.join(species_list) + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('species_labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=val_species_list,dtype=dt,compression='gzip')

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

# species_list = ['sCer','cEleg','Mouse','Human']
species_list = ['sCer','sBoul','sArb','sEub']

seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
sp in species_list]

# combine_species_allgenes(species_list,upstream_length,promoter_length)
# combine_species(species_list,upstream_length,promoter_length,window_step,3)

# seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
# sp in species_list]

# # read in promoter sequences and convert to one-hot encoding sequences
# seq_dicts = [read_fasta_seq(f) for f in seq_files]

def rewriteHDF5(h5_file,dir_name):

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('species_label',data=f['labels'][:],dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')
	g.create_dataset('species',data=f['species'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

dir_name = 'data/h5datasets/new2_all4/'
rewriteHDF5('train.h5',dir_name)
rewriteHDF5('validation.h5',dir_name)

dir_name = 'data/h5datasets/new2_Sac4/'
rewriteHDF5('train.h5',dir_name)
rewriteHDF5('validation.h5',dir_name)

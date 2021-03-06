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

def combine_species(species_list,upstream_length,promoter_length,window_step,sampling_method=1,random_sample=False):
	'''combines sequence data from all species into one large numpy array,
	creates species labels for all sequences in the array, and writes the 
	data to H5 datasets'''

	seq_files = ['data/my_promoters/' + sp + str(upstream_length) + '.fa.txt' for \
		sp in species_list]

	print('reading in sequences...')
    # read in promoter sequences and convert to one-hot encoding sequences
	seq_dicts = [read_fasta_seq(f) for f in seq_files]

	if sampling_method <= 2: 
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
		trainIdx = allIdx[0:int(0.9*len(mat))]
		valIdx = allIdx[int(0.9*len(mat)):]

		train_genes = [gene_list[i] for i in trainIdx]
		train_dat = np.array([mat[i] for i in trainIdx])
		train_labels = np.array([sp_labels[i] for i in trainIdx])

		validation_genes = [gene_list[i] for i in valIdx]
		validation_dat = np.array([mat[i] for i in valIdx])
		validation_labels = np.array([sp_labels[i] for i in valIdx])

	else: # ensure all genes are represented! (different # of windows/gene/species)
		print('creating windows...')
		gene_windows = [getallWindows(seq_dict,promoter_length,window_step,\
			concat=False,num_windows=5) for seq_dict in seq_dicts]
		windows = [wind for wind,gene in gene_windows]
		gene_list = [[gene for gene in sorted(seq_dict)] for seq_dict in seq_dicts]

		# determine number of total windows per species
		min_datsize = min([len(gene_window[1]) for gene_window in gene_windows])

		# determine number of windows per gene to select for each species
		train_num_windows_per_gene = [max(1,min(10,int(float(min_datsize)/len(seq_dict.keys())))) \
			for seq_dict in seq_dicts]
		val_num_windows_per_gene = [max(1,int(float(min_datsize)/len(seq_dict.keys()))) \
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
			[:,np.random.choice(range(windows[i].shape[1]),train_num_windows_per_gene[i],\
			replace=False)]) for i in range(len(windows))]
		validation_windows = [np.concatenate(windows[i][valIdx[i],:]\
			[:,np.random.choice(range(windows[i].shape[1]),val_num_windows_per_gene[i],\
			replace=False)]) for i in range(len(windows))]

		print('creating labels...')
		# create species labels for training and validation data
		num_windows_per_species_train = [window_set.shape[0] for \
			window_set in train_windows]
		num_windows_per_species_val = [window_set.shape[0] for \
			window_set in validation_windows]

		# create labels for randomly generated windows (set # of random windows
		# to be equal to # of windows in 1st species, i.e. # of windows in index 0)
		if random_sample:
			random_trainsize = num_windows_per_species_train[0]
			random_valsize = num_windows_per_species_val[0]

			num_windows_per_species_train.append(random_trainsize)
			num_windows_per_species_val.append(random_valsize)

			species_list.append('random')

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
			train_geneset = [[gene_list[i][j]]*train_num_windows_per_gene[i] for j in trainIdx[i]]
			train_genes.extend([gene for gene_rep in train_geneset for gene in gene_rep])

			val_geneset = [[gene_list[i][j]]*val_num_windows_per_gene[i] for j in valIdx[i]]
			validation_genes.extend([gene for gene_rep in val_geneset for gene in gene_rep])

		# aggregate training and validation data into large matrix
		train_dat = np.concatenate(train_windows)
		validation_dat = np.concatenate(validation_windows)

		# include randomly shuffled DNA sequences in the training, validation dataset
		if random_sample:
			print('shuffling to create random sequences...')
			random_train = train_dat[np.random.choice(range(train_dat.shape[0]),\
				random_trainsize)]
			random_train = np.array([shuffle_onehotseq(random_train[i]) for \
				i in range(random_train.shape[0])])
			train_dat = np.concatenate([train_dat,random_train])
			train_genes.extend(['random' for i in range(random_trainsize)])

			# random_val = validation_dat[np.random.choice(range(validation_dat.shape[0]),\
			# 	random_valsize)]
			# random_val = np.array([shuffle_onehotseq(random_val[i]) for \
			# 	i in range(random_val.shape[0])])
			# validation_dat = np.concatenate([validation_dat,random_val])
			# validation_genes.extend(['random' for i in range(random_valsize)])

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

	# create species list
	allgenes_species_list = [species_list[i] for i in range(len(species_list)) \
		for j in range(num_windows[i])]

	print('writing...')
	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File('+'.join(species_list) + '.all.h5','w')
	f.create_dataset('dnaseq',data=mat,dtype='f',compression='gzip')
	f.create_dataset('species_labels',data=sp_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=gene_list,dtype=dt,compression='gzip')
	f.create_dataset('species',data=allgenes_species_list,dtype=dt,compression='gzip')
	f.close()

##############################################################################

def filterDatabySpeciesLabel(h5_file,outfile,inds_to_keep):
	'''identifies the data stored in an H5 file that correspond to an inputted 
	set of desired label indices and writes this filtered data to another H5 
	file (Ex: include only dnaseq/genes from species labels with 1s at indices
	0 and 1)'''

	f = h5py.File(h5_file,'r')
	g = h5py.File(outfile,'w')

	dt = h5py.special_dtype(vlen=unicode)

	dataIdx = np.array([i for i in range(f['species_labels'].shape[0]) if \
		sum(f['species_labels'][i][np.array(inds_to_keep)]) > 0])

	g.create_dataset('dnaseq',data=f['dnaseq'][:][dataIdx],dtype='f',compression='gzip')
	g.create_dataset('species_labels',data=f['species_labels'][:][dataIdx],dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:][dataIdx],dtype=dt,compression='gzip')
	g.create_dataset('species',data=f['species'][:][dataIdx],dtype=dt,compression='gzip')

	f.close()
	g.close()

def rewriteHDF5(h5_file,dir_name):

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('species_labels',data=f['labels'][:],dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')
	g.create_dataset('species',data=f['species'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

def splitHDF(infile,outfile_name,floatkey_list,unicodekey_list):
	'''takes a dataset in HDF5 file and splits it into a training set and 
	validation set'''

	f = h5py.File(infile,'r')
	datsize = f[f.keys()[0]].shape[0]
	trainIdx = np.random.choice(range(datsize),int(0.8*datsize))
	valIdx = np.array([i for i in range(datsize) if i not in trainIdx])

	trainFile = h5py.File(outfile_name + 'train.h5','w')
	valFile = h5py.File(outfile_name + 'validation.h5','w')

	for key in floatkey_list:
		trainFile.create_dataset(key,data=f[key][:][trainIdx],dtype='f',\
			compression='gzip')
		valFile.create_dataset(key,data=f[key][:][valIdx],dtype='f',\
			compression='gzip')

	dt = h5py.special_dtype(vlen=unicode)

	for key in unicodekey_list:
		trainFile.create_dataset(key,data=f[key][:][trainIdx],dtype=dt,\
			compression='gzip')
		valFile.create_dataset(key,data=f[key][:][valIdx],dtype=dt,\
			compression='gzip')

	trainFile.close()
	valFile.close()

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


def rewriteHDF5_binary(h5_file,dir_name): # multi-label binary encoding

	f = h5py.File(dir_name + h5_file,'r')
	g = h5py.File(dir_name + h5_file + '_new','w')

	dt = h5py.special_dtype(vlen=unicode)

	species_labels = convertLabels(f['species_labels'][:])

	g.create_dataset('dnaseq',data=f['dnaseq'][:],dtype='f',compression='gzip')
	g.create_dataset('species_labels',data=species_labels,dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:],dtype=dt,compression='gzip')
	g.create_dataset('species',data=f['species'][:],dtype=dt,compression='gzip')

	f.close()
	g.close()

def evenHDF5(h5_file,dir_name,num_samples):
	'''takes as input an HDF5 file containing species information and samples
	from the data to generate a subset of the original file with an even number
	of species represented'''

	f = h5py.File(dir_name + h5_file,'r')
	newfile_name = h5_file.split('.')[0] + '_even.h5'
	g = h5py.File(dir_name + newfile_name,'w')

	labelIdx_list = np.argmax(f['species_labels'][:],axis=1)
	sampleIdx = []

	count_dict = {labelIdx: 0 for labelIdx in list(set(list(labelIdx_list)))}
	i = 0
	while sum(np.array(count_dict.values()) == num_samples) < f['species_labels'].shape[1]:
		labelIdx = labelIdx_list[i]
		if count_dict[labelIdx] < num_samples:
			sampleIdx.append(i)
			count_dict[labelIdx] += 1
		i += 1

	sampleIdx = np.array(sampleIdx)

	dt = h5py.special_dtype(vlen=unicode)

	g.create_dataset('dnaseq',data=f['dnaseq'][:][sampleIdx],\
		dtype='f',compression='gzip')
	g.create_dataset('species_labels',data=f['species_labels'][:][sampleIdx],\
		dtype='f',compression='gzip')
	g.create_dataset('genes',data=f['genes'][:][sampleIdx],\
		dtype=dt,compression='gzip')
	g.create_dataset('species',data=f['species'][:][sampleIdx],\
		dtype=dt,compression='gzip')

	f.close()
	g.close()

def splitGenesHDF5(h5_file,geneFile,species,outfile):
	'''takes an existing HDF5 file, filters out a specified species, and creates
	a training and a validation dataset with labels specifying whether or not 
	each data point belongs to a given set of genes'''

	# get gene list
	f = open(geneFile,'r')
	reader = csv.reader(f,delimiter='\t')
	chosenGenes = [line[0] for line in reader]
	f.close()

	# get indices of data points associated with gene list
	f = h5py.File(h5_file,'r')
	h5_genes = f['genes'][:]
	h5_species = f['species'][:]

	geneIdx = [i for i in range(h5_genes.shape[0]) if str(h5_genes[i]) in chosenGenes and \
		str(h5_species[i])==species]
	nongeneIdx = [i for i in range(h5_genes.shape[0]) if str(h5_genes[i]) not in chosenGenes and \
		str(h5_species[i])==species]
	nongeneIdx = list(np.random.choice(nongeneIdx,size=len(geneIdx),replace=False))

	allIdx = np.array(geneIdx + nongeneIdx)

	dat = f['dnaseq'][:][allIdx]
	labels = getLabels([len(geneIdx),len(nongeneIdx)])
	gene_list = h5_genes[allIdx]
	species_list = h5_species[allIdx]
	f.close()

	# split into training and validation dataset
	allIdx = range(dat.shape[0])
	np.random.shuffle(allIdx)
	trainIdx = allIdx[0:int(0.8*len(allIdx))]
	valIdx = allIdx[int(0.8*len(allIdx)):]

	# training data
	train_dat = dat[trainIdx]
	train_labels = labels[trainIdx]
	train_genes = gene_list[trainIdx]
	train_species = species_list[trainIdx]

	# validation data
	validation_dat = dat[valIdx]
	validation_labels = labels[valIdx]
	validation_genes = gene_list[valIdx]
	validation_species = species_list[valIdx]

	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File(outfile + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=train_species,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File(outfile + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=validation_species,dtype=dt,compression='gzip')

	f.close()

def splitGenesSpeciesHDF5(h5_file,geneFile_list,chosenSpecies,outfile):
	'''takes an existing HDF5 file, filters out a specified species, and creates
	a training and a validation dataset with labels specifying whether or not 
	each data point belongs to a given set of genes'''

	# get gene lists
	chosenGenes = []
	for i in range(len(geneFile_list)):
		geneFile = geneFile_list[i]
		species = chosenSpecies[i]
		f = open(geneFile,'r')
		reader = csv.reader(f,delimiter='\t')
		chosenGenes_list = [(line[0],species) for line in reader]
		chosenGenes.extend(chosenGenes_list)
		f.close()

	# get indices of data points associated with gene list
	f = h5py.File(h5_file,'r')
	h5_genes = f['genes'][:]
	h5_species = f['species'][:]

	geneIdx = [i for i in range(h5_genes.shape[0]) if \
		(str(h5_genes[i]),str(h5_species[i])) in chosenGenes]

	allIdx = geneIdx

	dat = f['dnaseq'][:][allIdx]
	gene_list = h5_genes[allIdx]
	species_list = h5_species[allIdx]
	f.close()

	# create labels
	label_set = np.identity(len(chosenSpecies),dtype=np.float32)
	label_dict = {chosenSpecies[i]: label_set[i] for i in range(len(chosenSpecies))}
	labels = np.array([label_dict[species] for species in species_list])
	
	# split into training and validation dataset
	allIdx = range(dat.shape[0])
	np.random.shuffle(allIdx)
	trainIdx = allIdx[0:int(0.8*len(allIdx))]
	valIdx = allIdx[int(0.8*len(allIdx)):]

	# training data
	train_dat = dat[trainIdx]
	train_labels = labels[trainIdx]
	train_genes = gene_list[trainIdx]
	train_species = species_list[trainIdx]

	# validation data
	validation_dat = dat[valIdx]
	validation_labels = labels[valIdx]
	validation_genes = gene_list[valIdx]
	validation_species = species_list[valIdx]

	dt = h5py.special_dtype(vlen=unicode)

	f = h5py.File(outfile + '.train.h5','w')
	f.create_dataset('dnaseq',data=train_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=train_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=train_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=train_species,dtype=dt,compression='gzip')
	f.close()

	f = h5py.File(outfile + '.validation.h5','w')
	f.create_dataset('dnaseq',data=validation_dat,dtype='f',compression='gzip')
	f.create_dataset('labels',data=validation_labels,dtype='f',compression='gzip')
	f.create_dataset('genes',data=validation_genes,dtype=dt,compression='gzip')
	f.create_dataset('species',data=validation_species,dtype=dt,compression='gzip')

	f.close()

def filterReps():
	'''takes a file containing representations of data and 


h5_file = 'data/h5datasets/all10bin/all.h5'
geneFile1 = 'results/clustering/RShiny/selectedPointsHuman.txt'
geneFile2 = 'results/clustering/RShiny/selectedPointsMouse.txt'
geneFile_list = [geneFile1,geneFile2]
species_list = ['Human','Mouse']

outfile = 'Clusters'

splitGenesSpeciesHDF5(h5_file,geneFile_list,species_list,outfile)

# evenHDF5('validation.h5','data/h5datasets/all10/',1000)
# rewriteHDF5_binary('all.h5','data/h5datasets/all10/')

# dir_name = 'data/h5datasets/all10/'
# rewriteHDF5_binary('train.h5',dir_name)
# rewriteHDF5_binary('validation.h5',dir_name)

# floatkey_list = ['dnaseq','labels']
# unicodekey_list = ['genes']
# splitHDF('data/h5datasets/certain_Human/Human_certain.h5','Human',floatkey_list,unicodekey_list)
# splitHDF('data/h5datasets/certain_Mouse/Mouse_certain.h5','Mouse',floatkey_list,unicodekey_list)

# h5_file = 'data/h5datasets/all4_unmasked/train.h5'
# outfile = 'data/h5datasets/all4_unmasked/train_sCer_cEleg.h5'
# inds_to_keep = [0,1]
# filterDatabySpeciesLabel(h5_file,outfile,inds_to_keep)

# dir_name = 'data/h5datasets/new2_all4/'
# rewriteHDF5('train.h5',dir_name)
# rewriteHDF5('validation.h5',dir_name)

# dir_name = 'data/h5datasets/Mouse_Human/'
# rewriteHDF5('train.h5',dir_name)
# rewriteHDF5('validation.h5',dir_name)

# upstream_length = 1000
# promoter_length = 500
# window_step = 100

species_list = ['sCer','cEleg','Mouse','Human','sPom','Zebrafish','dMelan','Chicken','aThal','Lizard']
# combine_species_allgenes(species_list,upstream_length,promoter_length)

# # species_list = ['Mouse','Human']
# species_list = ['sCer','cEleg','Mouse','Human','sPom','Zebrafish','dMelan','Chicken','aThal','Lizard']
# combine_species(species_list,upstream_length,promoter_length,window_step,2,False)

# species_list = ['sCer','sPom']
# combine_species(species_list,upstream_length,promoter_length,window_step,3,True)

# species_list = ['sCer','cEleg','Mouse','Human']
# combine_species(species_list,upstream_length,promoter_length,window_step,3,True)

# species_list = ['sCer','sBoul','sArb','sEub']
# combine_species(species_list,upstream_length,promoter_length,window_step,3,True)


import numpy as np
import csv
import h5py

from model import *

def readFASTA(file_name,gene_list=None):
    '''reads in a file of promoter sequences in FASTA format and returns a 
    dictionary of one-hot encoded DNA sequences indexed by gene name'''

    seq_dict = {}

    with open(file_name,'r') as f:
        reader = csv.reader(f,delimiter=',')
        gene_name = reader.next()[0].split('>')[1].strip()
        # gene_name = gene_name.split('_')[0]
        seq = ''
        for line in reader:
	        if '>' in line[0]:
	            if len(seq) > 0:
	            	print(len(seq_dict))
	                seq_dict[gene_name.lower()] = \
	                    one_hot_encode_sequence(seq)

	            # read in header of new sequence
	            gene_name = line[0].split('>')[1].strip()
	            seq = ''
	        else:
	            seq += line[0]

    return seq_dict


def writeDNA2HDF5(seq_file,outfile):
    '''reads in a FASTA file, one-hot encodes the sequences, and writes the
    one-hot encoded sequences to an HDF5 file'''

    seq_dict = readFASTA(seq_file)

    mat = np.array([seq_dict[key] for key in seq_dict])

    f = h5py.File(outfile,'w')
    f.create_dataset('dnaseq',data=mat,dtype='f',compression='gzip')
    f.close()

def writeDNAplusLabels2HDF5(seq_file1,seq_file2,outfile_name,real_file=False,balance=False):
	'''reads in two FASTA files, one-hot encodes their sequences, and creates 
	training and validation datasets with the sequences randomly shuffled with
	labels indicating the file to which each of them belong'''

	seq_dicts = [readFASTA(f) for f in [seq_file1,seq_file2]]

	# use same number of genes/regions from each FASTA file
	if balance:
		min_datsize = min([len(seq_dict) for seq_dict in seq_dicts])
		seq_dicts = [{key: seq_dict[key] for key in seq_dict.keys()[0:min_datsize]} for \
			seq_dict in seq_dicts]

	mat = np.concatenate([np.array([seq_dict[key] for key in seq_dict]) \
		for seq_dict in seq_dicts])

	labs = np.array([[1.,0.],[0.,1.]])
	all_labels = np.repeat(labs,[len(seq_dicts[0]),len(seq_dicts[1])],axis=0)

	# shuffle data and labels
	shuffleIdx = range(mat.shape[0])
	np.random.shuffle(shuffleIdx)
	mat = mat[shuffleIdx]
	all_labels = all_labels[shuffleIdx]

	split_pt = int(0.8*mat.shape[0])

	# create training data
	f = h5py.File(outfile_name + 'train.h5','w')
	f.create_dataset('dnaseq',data=mat[0:split_pt],dtype='f',compression='gzip')
	f.create_dataset('labels',data=all_labels[0:split_pt],compression='gzip')
	f.close()	

	# create training data
	f = h5py.File(outfile_name + 'validation.h5','w')
	f.create_dataset('dnaseq',data=mat[split_pt:],dtype='f',compression='gzip')
	f.create_dataset('labels',data=all_labels[split_pt:],compression='gzip')
	f.close()	

seq_file1 = 'data/my_promoters/Human200.fa.txt' # 'data/simulated_dna/TAL1.fa'
seq_file2 = 'data/simulated_dna/background.fa'

writeDNAplusLabels2HDF5(seq_file1,seq_file2,'Human200_background',balance=True)
# writeDNA2HDF5('data/simulated_dna/TAL1.fa','data/simulated_dna/TAL1.h5')

# writeDNA2HDF5('data/simulated_dna/background.fa','data/simulated_dna/background.h5')
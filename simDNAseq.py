import numpy as np
import csv
import h5py

from model import *

def readFASTA(file_name,gene_list=None):
    '''reads in a file of promoter sequences in FASTA format and returns a 
    dictionary of one-hot encoded DNA sequences indexed by gene name'''

    seq_dict = {}

    with open(file_name,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        gene_name = reader.next()[0][1:]
        # gene_name = gene_name.split('_')[0]
        seq = ''
        for line in reader:
            if '>' in line[0]:
                # one hot encode completed sequence if sufficiently low 
                # number of "N"
                if len(seq) > 0:
                    seq_dict[gene_name.lower()] = \
                        one_hot_encode_sequence(seq)

                # read in header of new sequence
                gene_name = line[0][1:]
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

def writeDNAplusLabels2HDF5(seq_file1,seq_file2,outfile_name):
	'''reads in two FASTA files, one-hot encodes their sequences, and creates 
	training and validation datasets with the sequences randomly shuffled with
	labels indicating the file to which each of them belong'''

	seq_dicts = [readFASTA(f) for f in [seq_file1,seq_file2]]

	mat = np.concatenate([np.array([seq_dict[key] for key in seq_dict]) \
		for seq_dict in seq_dicts])

	labs = np.array([[1.,0.],[0.,1.]])
	all_labels = np.repeat(labs,[len(seq_dicts[0]),len(seq_dicts[1])])

	# create training data
    f = h5py.File(outfile,'w')
    f.create_dataset('dnaseq',data=mat,dtype='f',compression='gzip')
    f.close()	


writeDNA2HDF5('data/simulated_dna/TAL1.fa','data/simulated_dna/TAL1.h5')

writeDNA2HDF5('data/simulated_dna/background.fa','data/simulated_dna/background.h5')

# seq_dict = readFASTA('data/simulated_dna/TAL1.fa')
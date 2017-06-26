from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import itertools
import csv
import os.path
import gzip

from keras import backend as K
from keras.objectives import kullback_leibler_divergence
from sklearn.metrics import hamming_loss

def rand_data(N):
    mat = np.array([one_hot_encode_sequence(np.random.choice(['a','t','c','g'],500)) for i in range(N)])
    return mat

### SEQUENCE DATA PREPROCESSING ###############################################
def one_hot_encode_sequence(seq):
    '''converts a DNA sequence into its corresponding one-hot encoding 
    representation'''

    seq = [x.lower() for x in seq]

    result = [[],[],[],[]]

    bps = ['a','t','c','g']
    for i in range(len(seq)):
        if seq[i] == 'n':
            for j in range(len(bps)):
                result[j].append(np.array([0.25]))
        else:
            for j in range(len(bps)):
                if seq[i] == bps[j]:
                    result[j].append(np.array([1.]))
                else:
                    result[j].append(np.array([0.]))
    result = [np.array(x) for x in result]
    if len(result[0]) < 50:
        print(seq)
    return np.array(result)

def read_fasta_seq(file_name):
    '''reads in a file of promoter sequences in FASTA format and returns a 
    dictionary of one-hot encoded DNA sequences indexed by gene name'''

    seq_dict = {}

    with open(file_name,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        gene_name = reader.next()[1]
        gene_name = gene_name.split('.')[0]
        # gene_name = gene_name.split('_')[0]
        seq = ''
        for line in reader:
            if len(line) > 1:
                # one hot encode completed sequence if sufficiently low 
                # number of "N"
                if seq.count('N') <= 0.1*len(seq) and len(seq) > 0:
                    seq_dict[gene_name] = one_hot_encode_sequence(seq)

                # read in header of new sequence
                gene_name = line[1]
                gene_name = gene_name.split('.')[0]
                # gene_name = gene_name.split('_')[0]
                seq = ''
            else:
                seq += line[0]

    return seq_dict

def read_gtf(gtf_file):
    '''reads in a GTF file and returns a dictionary of relevant gene info
    (i.e., gene name, +/- strand, start, end) indexed by chromosome number'''

    with gzip.open(gtf_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')

        annot_dict = {}
        a = []
        for line in reader:
            if len(line) > 2:
                if line[2] == 'gene':
                    annot = line[8].split(';')[1]
                    if 'gene_name' in annot:
                        chrno = 'chr'+line[0]
                        if chrno not in annot_dict:
                            annot_dict[chrno] = {}
                        gene = annot.split('"')[1]
                        annot_dict[chrno][gene] = {}                    
                        annot_dict[chrno][gene]['start'] = int(line[3])
                        annot_dict[chrno][gene]['end'] = int(line[4])
                        annot_dict[chrno][gene]['strand'] = line[6]
                                        
    return annot_dict

def reverse_complement(seq):
    '''outputs the reverse complement of an inputted DNA sequence'''

    revcomp = ""

    for s in reversed(seq):
        if s == "A":
            revcomp += "T"
        elif s == "C":
            revcomp += "G"
        elif s == "G":
            revcomp += "C"
        elif s == "T":
            revcomp += "A"
        else:
            revcomp += "N"
    return revcomp

def getchrSequences(annot_dict,chrno,chrdat_dir,seq_length):
    '''gets the DNA sequences for all genes in a particular chromosome by 
    reading through a FASTA file containing the sequence for that chromosome
    and converts them to one-hot encoded sequences

    (NOTE: the FASTA file must be named chrX.fa if looking for chromsome X'''

    # dictionary of region sequences indexed by gene name
    region_dict = {}

    # get entire chromosome sequence
    with gzip.open(chrdat_dir + '/'+ chrno + '.fa.gz','r') as f:
        reader = csv.reader(f,delimiter='\t')
        reader.next() # ignore header
        sequence = ''
        for line in reader:
            sequence += line[0]

    for gene in annot_dict[chrno]:
        if annot_dict[chrno][gene]['strand'] == '+':
            start = annot_dict[chrno][gene]['start']
            region_seq = sequence[start-1-seq_length:start-1]
            if len(region_seq) == seq_length:
                region_dict[gene] = one_hot_encode_sequence(region_seq)

        elif annot_dict[chrno][gene]['strand'] == '-':
            start = annot_dict[chrno][gene]['end']
            region_seq = sequence[start-1:start-1+seq_length]
            if len(region_seq) == seq_length:
                region_dict[gene] = \
                    one_hot_encode_sequence(reverse_complement(region_seq))


    return region_dict

def readSequences(annot_dict,chrdat_dir,seq_length):
    '''uses an annotation dictionary and genome-wide FASTA files to obtain
    region sequences for all genes in the genome'''

    seq_dict = {}

    for chrno in annot_dict:
        print(chrno)
        if os.path.isfile(chrdat_dir + '/' + chrno + '.fa.gz'):
            seq_dict.update(getchrSequences(annot_dict,chrno,chrdat_dir,seq_length))

    return seq_dict

def seqdict2mat(seq_dict):
    '''converts a dictionary of one-hot encoded DNA sequences into an array
    of one-hot encoded DNA sequences'''

    mat = []

    sorted_genelist = sorted(seq_dict.keys())
    for i in range(len(seq_dict)):
        mat.append(seq_dict[sorted_genelist[i]])

    return np.array(mat)

def getLabels(catcount_list):
    '''creates one-hot encoded labels for species/GO terms represented using an
    inputted list of window counts for each species/GO term'''

    label_set = [np.zeros(len(catcount_list),dtype=np.float32) for i \
        in catcount_list]
    for i in range(len(label_set)):
        label_set[i][i] = 1.

    labels = []
    for i in range(len(catcount_list)):
        for num_windows in range(catcount_list[i]):
            labels.append(label_set[i])

    return np.array(labels)


### DNA SEQUENCE SLIDING WINDOW ###############################################

def getrandWindow(seq_dict,window_size):
    '''selects a random window of inputted size for all DNA regions represented
    in an inputted dictionary of sequences sorted alphabetically by DNA region
    name/gene name'''

    all_windows = []
    for reg in sorted(seq_dict.keys()):
        onehot_seq = seq_dict[reg]
        window = [np.zeros(window_size,dtype=np.float32) for i in range(4)]
        randIdx = np.random.choice(range(onehot_seq.shape[1]-window_size),1)[0]
        for j in range(4): # iterate through 4 bp arrays
            window[j] = onehot_seq[j][randIdx:randIdx+window_size]
        window = np.array(window)
        all_windows.append(window)
    return np.array(all_windows)

def slideWindow(onehot_seq,window_size,step_size):
    '''takes as input a one-hot encoded DNA sequence and outputs windows of a
    given size and generated from a given step size across the sequence'''

    all_windows = []

    for i in range(0,len(onehot_seq[0])-step_size,step_size): # iterate through all steps
        window = [np.zeros(window_size,dtype=np.float32) for i in range(len(onehot_seq))] # empty windows
        for j in range(len(onehot_seq)): # iterate through 4 bp arrays
            # add window for each bp
            window[j] = onehot_seq[j][i:i+window_size]

        window = np.array(window)
        all_windows.append(window)

    return all_windows

def getallWindows(seq_dict,window_size,step_size):
    '''uses the sliding window approach to get windows for all DNA regions
    represented in an inputted dictionary of sequences sorted alphabetically
    by DNA region name/gene name'''

    all_dat = []
    gene_list = []

    for reg in sorted(seq_dict):
        print(reg)
        reg_windows = slideWindow(seq_dict[reg],window_size,step_size)
        all_dat.append(reg_windows)
        gene_list.extend([reg for i in range(len(reg_windows))])

    all_dat = [window for regwind in all_dat for window in regwind]

    return np.array(all_dat), gene_list

def multilabelWindows(seq_dicts,geneterm_dict,ontology_list,window_size,step_size):
    '''generates windows for all sequences represented in both the sequence
    dictionaries and the gene ontology dictionaries; also returns the
    corresponding non-mutually exclusive labels to each of the windows'''

    # assign label index to each GO term in the ontology list
    GOinds_dict = {}
    for i in range(len(ontology_list)):
        GOterm = ontology_list[i]
        GOinds_dict[GOterm] = i

    gene_count = 0

    allWindows = []
    allLabels = []
    for species_gene_name in geneterm_dict:

        name = species_gene_name.split('~')
        species = name[0]
        gene = name[1]
        # get sequences & create windows, labels if gene in sequence dictionary
        if gene in seq_dicts[species] or gene.lower() in seq_dicts[species] or \
            gene.upper() in seq_dicts[species]:

            # create windows
            if gene in seq_dicts[species]:
                windows = slideWindow(seq_dicts[species][gene],\
                    window_size,step_size)
                allWindows.extend(windows)
            elif gene.lower() in seq_dicts[species]:
                windows = slideWindow(seq_dicts[species][gene.lower()],\
                    window_size,step_size)
                allWindows.extend(windows)
            elif gene.upper() in seq_dicts[species]:
                windows = slideWindow(seq_dicts[species][gene.upper()],\
                    window_size,step_size)  
                allWindows.extend(windows) 

            gene_count += 1

            # create labels for all windows
            label = np.zeros(len(ontology_list),dtype=np.float32)
            for GOterm in geneterm_dict[species_gene_name]:
                label[GOinds_dict[GOterm]] = 1.

            # replicate labels for all windows created for the gene/region
            allLabels.extend([label for i in range(len(windows))])

    print(str(gene_count) + '/' + str(len(geneterm_dict)))
    return (np.array(allWindows),np.array(allLabels))


### PATHWAY/ONTOLOGY DATA PREPROCESSING #######################################
def pathway_dict(file_name):
    '''reads in a KEGG pathway data file and creates a dictionary of gene sets
    indexed by pathway name'''

    pw_dict = {}

    f = open(file_name,'r')
    reader = csv.reader(f,delimiter='\t')
    for line in reader:
        pw_name = line[0]
        genes = line[2:]
        pw_dict[pw_name] = genes

    return pw_dict

def reverse_pathway_dict(pw_dict):
    '''reverses the dictionary of gene sets indexed by pathway to generate a 
    dictionary of pathways indexed by gene'''

    gene_dict = {}

    for pw in pw_dict:
        for gene in pw_dict[pw]:
            if gene not in gene_dict:
                gene_dict[gene] = []
            gene_dict[gene].append(pw)

    return gene_dict

def pathway_labels(seq_dict,gene_dict,pw_dict):
    '''creates pathway labels for each sequence represented in
        the sequence dictionaries (uses multi-labeling! - due to possibility of
        > 1 pathways/gene'''

    # index all pathways represented in the dictionary of pathways
    sorted_pathways = sorted(pw_dict.keys())
    pwIdx_dict = {pw: Idx for Idx,pw in enumerate(sorted_pathways)}

    # create labels for each gene
    lab_list = []
    for gene in sorted(seq_dict):
        lab = np.zeros(len(pw_dict.keys()),dtype=np.float32)
        for pw in gene_dict[gene]:
            lab[pwIdx_dict[pw]] = 1.
        lab_list.append(lab)

    return lab_list

def getGOterms(infile,species_list):
    '''reads in a file downloaded from AmiGO2 and returns a dictionary of 
    genes indexed by species name'''

    f = open(infile,'r')
    reader = csv.reader(f,delimiter='\t')

    GO_dict = {species:[] for species in species_list}
    for line in reader:
        for species in species_list:
            if species in line[1]:
                GO_dict[species].append(line[2])

    for species in GO_dict:
        GO_dict[species] = list(set(GO_dict[species]))

    f.close()

    return GO_dict

def combineGOseq(seq_dicts,GO_dicts):
    '''integrates GO term-gene dictionaries with gene-sequence dictionaries
    to get a dictionary of dictionaries of sequences indexed by species-gene 
    indexed by ontology names also returns a dictionary of ontologies indexed 
    by species-gene names'''

    GOseq_dicts = {}
    geneterm_dict = {}
    for GOterm in GO_dicts.keys():
        GO_dict = GO_dicts[GOterm]
        GOseq_dict = {}
        for species in GO_dict:
            for gene in GO_dict[species]:
                if species + '-' + gene not in geneterm_dict:
                    geneterm_dict[species + '-' + gene] = []
                geneterm_dict[species + '-' + gene].append(GOterm)
                if gene in seq_dicts[species]:
                    GOseq_dict[species + '-' + gene] = seq_dicts[species][gene]
                elif gene.lower() in seq_dicts[species]:
                    GOseq_dict[species + '-' + gene] = \
                        seq_dicts[species][gene.lower()]
        GOseq_dicts[GOterm] = GOseq_dict
    return (GOseq_dicts,geneterm_dict)

def geneGO_dict(GO_dicts):
    '''returns a dictionary of GO term lists indexed by species+gene name by 
    rearranging contents of the inputted dictionary of gene names indexed by
    species and GO terms'''

    geneterm_dict = {}
    for GOterm in GO_dicts.keys():
        for species in GO_dicts[GOterm]:
            for gene in GO_dicts[GOterm][species]:
                if species + '~' + gene.lower() not in geneterm_dict:
                    geneterm_dict[species + '~' + gene.lower()] = []
                geneterm_dict[species + '~' + gene.lower()].append(GOterm)
    return geneterm_dict
    
### ACCURACY AND LOSS FUNCTIONS ###############################################
def calc_hamming_loss(y_true,y_pred):
    '''calculates the Hamming loss using the scikit-learn implementation'''

    return hamming_loss(y_true,np.array(y_pred))


### MISCELLANEOUS #############################################################

def batcher(num_examples,batch_size):

    iterable = xrange(0,num_examples)
    for i in itertools.cycle(iterable):
        yield (i,i+batch_size)


### DATA OBJECT ###############################################################

class TrainData(object):
    '''training data object'''

    def __init__(self,h5_file,batch_size):
        self.h5_file = h5_file
        self.batch_size = batch_size

    def batcher(self):
        iterable = xrange(0,self.h5_file.values()[0].shape[0] - \
            self.batch_size,self.batch_size)
        for i in itertools.cycle(iterable):
            yield {key: val[i:i+self.batch_size] for key,val in \
                self.h5_file.items()}

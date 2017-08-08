import csv
import os.path
import gzip
import h5py
import numpy as np

class Gene(object):
    def __init__(self,genename,chrno,start,end,strand):
        self.genename = genename
        self.chrno = chrno
        self.start = int(start)
        self.end = int(end)
        self.strand = strand

def readBED(BEDfile):
    '''reads in a GTF file and returns a dictionary of gene object lists containing 
    relevant gene info (i.e., chromosome number, gene name, +/- strand, 
    start, end) for each gene indexed by chromosome number'''

    annot_dict = {}
    with gzip.open(BEDfile,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        for line in reader:
            chrno = line[0]
            start = line[3] #line[1]
            end = line[4] #line[2]
            genename = line[8] #line[3]
            strand = line[6] # line[4]

            if chrno not in annot_dict:
                annot_dict[chrno] = []
            annot_dict[chrno].append(Gene(genename,chrno,start,end,strand))
                                        
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

def getchrSequences(annot_dict,chrno,chrdat_dir):
    '''gets the DNA sequences for all genes in a particular chromosome by 
    reading through a FASTA file containing the sequence for that chromosome

    (NOTE: the FASTA file must be named chrX.fa if looking for chromsome X'''

    # dictionary of region sequences indexed by gene name
    region_dict = {}

    # get entire chromosome sequence
    with gzip.open(chrdat_dir + '/chr'+ chrno + '.fa.gz','r') as f:
        reader = csv.reader(f,delimiter='\t')
        reader.next() # ignore header
        sequence = ''
        for line in reader:
            sequence += line[0]

    for geneObj in annot_dict[chrno]:
        region_seq = sequence[geneObj.start-1:geneObj.end-1]

        if geneObj.strand == '-':
            region_seq = reverse_complement(region_seq)

        region_dict[geneObj] = region_seq

    return region_dict

def readSequences(annot_dict,chrdat_dir):
    '''uses an annotation dictionary and genome-wide FASTA files to obtain
    region sequences for all genes in the genome'''

    seq_dict = {}

    for chrno in annot_dict:
        print(chrno)
        if os.path.isfile(chrdat_dir + '/chr' + chrno + '.fa.gz'):
            seq_dict.update(getchrSequences(annot_dict,chrno,chrdat_dir))
    return seq_dict

# def writeSequence(seq_dict,outfile_name):
#     '''writes the sequences for each indexed region/gene in the sequence 
#     dictionary to a file in FASTA format'''

#     f = open(outfile_name,'w')
#     writer = csv.writer(f,delimiter=' ')

#     for region in seq_dict:
#         writer.writerow(['>',region])
#         for i in range(0,len(seq_dict[region]),80):
#             writer.writerow([seq_dict[region][i:i+80]])

#     f.close()

def one_hot_encode_sequence(seq):
    '''converts a DNA sequence into its corresponding one-hot encoding 
    representation'''

    seq = seq.lower()
    letterdict = {'a': [1, 0, 0, 0], 't': [0, 1, 0, 0], 'c': [0, 0, 1, 0],
                  'g': [0, 0, 0, 1], 'n': [0.25, 0.25, 0.25, 0.25]}
                  
    result = np.array([letterdict[x] for x in seq])

    return np.expand_dims(result.T,3)

def convertWriteSeq(seq_dict,outfile):
    '''converts the sequences represented in the inputted annotation dictionary
    to one-hot encoded sequences and writes the sequences along with their gene
    names to HDF5 files'''

    print(len(seq_dict))
    # convert dictionary of sequences to dictionary of one-hot encoded sequences
    onehotseq_dict = {geneObj: one_hot_encode_sequence(seq_dict[geneObj]) for \
        geneObj in seq_dict}

    # sort gene objects by gene name
    sorted_geneObj = sorted(seq_dict.keys(), key=lambda x: x.genename)

    # generate list of gene names and sequences in order to be written
    gene_list = []
    onehotSeqs = []
    for geneObj in sorted_geneObj:
        gene_list.append(geneObj.genename)
        onehotSeqs.append(onehotseq_dict[geneObj])

    gene_list = np.array(gene_list)
    onehotSeqs = np.array(onehotSeqs)

    dt = h5py.special_dtype(vlen=unicode)

    f = h5py.File(outfile,'w')
    f.create_dataset('dnaseq',data=onehotSeqs,dtype='f',compression='gzip')
    f.create_dataset('genes',data=gene_list,dtype=dt,compression='gzip')
    f.close()

def writeBEDSeqHDF5(BEDfile,genomedat_dir,outfile):
    '''takes a BED file and a directory containing FASTA files as input and
    writes the sequences in the BED file to an HDF5 file in one-hot encoded
    form'''

    annot_dict = readBED(BEDfile)
    seq_dict = readSequences(annot_dict,genomedat_dir)
    convertWriteSeq(seq_dict,outfile)

writeBEDSeqHDF5('data/annotation_files/boo.gtf.gz','data/genome_files/sBoul','blah.h5')
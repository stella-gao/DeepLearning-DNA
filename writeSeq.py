import csv
import os.path
import gzip

def read_gtf(gtf_file):
    '''reads in a GTF file and returns a dictionary of relevant gene info
    (i.e., gene name, +/- strand, start, end) indexed by chromosome number'''

    with gzip.open(gtf_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')

        annot_dict = {}

        for line in reader:
            if len(line) > 2:
                if line[2] == 'gene':
                    annotations = line[8].split(';')
                    for annot in annotations:                
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
                region_dict[gene] = region_seq

        elif annot_dict[chrno][gene]['strand'] == '-':
            start = annot_dict[chrno][gene]['end']
            region_seq = sequence[start-1:start-1+seq_length]
            if len(region_seq) == seq_length:
                region_dict[gene] = reverse_complement(region_seq)

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

def writeSequence(seq_dict,outfile_name):
    '''writes the sequences for each indexed region/gene in the sequence 
    dictionary to a file in FASTA format'''

    f = open(outfile_name,'w')
    writer = csv.writer(f,delimiter=' ')

    for region in seq_dict:
        writer.writerow(['>',region])
        for i in range(0,len(seq_dict[region]),80):
            writer.writerow([seq_dict[region][i:i+80]])

    f.close()

species = 'Human'
ann_file = 'data/annotation_files/' + species + '.gtf.gz'
genomedat_dir = 'data/genome_files/' + species
promoter_length = 1000

annot_dict = read_gtf(ann_file)
seq_dict = readSequences(annot_dict,genomedat_dir,promoter_length)
writeSequence(seq_dict,species + str(promoter_length) + '.fa.txt')

species = 'Mouse'
ann_file = 'data/annotation_files/' + species + '.gtf.gz'
genomedat_dir = 'data/genome_files/' + species
promoter_length = 1000

annot_dict = read_gtf(ann_file)
seq_dict = readSequences(annot_dict,genomedat_dir,promoter_length)
writeSequence(seq_dict,species + str(promoter_length) + '.fa.txt')

species = 'cEleg'
ann_file = 'data/annotation_files/' + species + '.gtf.gz'
genomedat_dir = 'data/genome_files/' + species
promoter_length = 1000

annot_dict = read_gtf(ann_file)
seq_dict = readSequences(annot_dict,genomedat_dir,promoter_length)
writeSequence(seq_dict,species + str(promoter_length) + '.fa.txt')

species = 'sCer'
ann_file = 'data/annotation_files/' + species + '.gtf.gz'
genomedat_dir = 'data/genome_files/' + species
promoter_length = 1000

annot_dict = read_gtf(ann_file)
seq_dict = readSequences(annot_dict,genomedat_dir,promoter_length)
writeSequence(seq_dict,species + str(promoter_length) + '.fa.txt')

species = 'sPom'
ann_file = 'data/annotation_files/' + species + '.gtf.gz'
genomedat_dir = 'data/genome_files/' + species
promoter_length = 1000

annot_dict = read_gtf(ann_file)
seq_dict = readSequences(annot_dict,genomedat_dir,promoter_length)
writeSequence(seq_dict,species + str(promoter_length) + '.fa.txt')

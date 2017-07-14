from model import *

def read_fasta_seq_raw(file_name,gene_list=None):
    '''reads in a file of promoter sequences in FASTA format and returns a 
    dictionary of DNA sequences indexed by gene name'''

    seq_dict = {}

    with open(file_name,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        gene_name = reader.next()[1]
        gene_name = gene_name.split('.')[0]
        # gene_name = gene_name.split('_')[0]
        seq = ''
        for line in reader:
            if len(line) > 1:
                # only include sequence if corresponding gene in gene list
                if gene_list:
                    if gene_name in gene_list or gene_name.lower() in \
                        gene_list or gene_name.upper() in gene_list:
                        if seq.count('N') <= 0.2*len(seq) and len(seq) > 0: 
                            seq_dict[gene_name.lower()] = seq
                else:
                    # one hot encode completed sequence if sufficiently low 
                    # number of "N"
                    if seq.count('N') <= 0.2*len(seq) and len(seq) > 0:
                        seq_dict[gene_name.lower()] = \
                            one_hot_encode_sequence(seq)

                # read in header of new sequence
                gene_name = line[1]
                gene_name = gene_name.split('.')[0]
                seq = ''
            else:
                seq += line[0]

    return seq_dict
    

def detectPattern(species,species_name,ontology_terms,pattern_list):
	'''returns a list of CpG island counts for promoters belonging to a set of
	GO terms for a given species'''

	ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]
	pattern_dict = {pattern: {GOterm: [] for GOterm in ontology_terms} for pattern in pattern_list}

	GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],[species_name]) \
	    for i in range(len(ontology_files))}

	gene_list = list(set([gene for GOterm in ontology_terms for gene in \
		GO_dicts[GOterm][species_name]]))

	seq_file = 'data/my_promoters/' + species + '1000.fa.txt' 
	seq_dict = read_fasta_seq_raw(seq_file,gene_list)

	for GOterm in GO_dicts:
		for pattern in pattern_list:
			for gene in GO_dicts[GOterm][species_name]:
				if gene.lower() in seq_dict:
					pattern_count = seq_dict[gene.lower()].count(pattern)
					if pattern_count > 0:
						pattern_dict[pattern][GOterm].append(pattern_count)

	return pattern_dict

# species = 'Human'
# species_name = 'sapien'
# ontology_terms = ['skeletal','multicell_dev']
# pattern_list = ['CG','CGCG','CGCGCG','CGCGCG','CGCGCGCG']

# pattern_dict = detectPattern(species,species_name,ontology_terms,pattern_list)

# ontology_files = ['data/GOterms/' + GOterm + '.txt' for GOterm in ontology_terms]
# pattern_dict = {pattern: {GOterm: [] for GOterm in ontology_terms} for pattern in pattern_list}
# GO_dicts = {ontology_terms[i]:getGOterms(ontology_files[i],[species_name]) \
#     for i in range(len(ontology_files))}

# gene_list = list(set([gene for GOterm in ontology_terms for gene in \
# 	GO_dicts[GOterm][species_name]]))

# seq_file = 'data/my_promoters/' + species + '1000.fa.txt' 
# seq_dict = read_fasta_seq_raw(seq_file,gene_list)

# for GOterm in GO_dicts:
# 	for pattern in pattern_list:
# 		for gene in GO_dicts[GOterm][species_name]:
# 			if gene.lower() in seq_dict:
# 				pattern_dict[pattern][GOterm].append(seq_dict[gene.lower()].count(pattern))

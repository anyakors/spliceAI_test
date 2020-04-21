import numpy as np
from Bio import SeqIO
import random
import os
import sys

def complementary(let):
	#A-T, C-G
	if let=='A':
		return 'T'
	if let=='T':
		return 'A'
	if let=='C':
		return 'G'
	if let=='G':
		return 'C'

def hot_encode_seq(let):
	#hot-encode the sequence, where "O" corresponds to zero-padded areas
    if let=='A':
        return([1,0,0,0])
    elif let=='T':
        return([0,1,0,0])
    elif let=='C':
        return([0,0,1,0])
    elif let=='G':
        return([0,0,0,1])
    elif let=='O':
        return([0,0,0,0])

def hot_encode_label(let):
	#hot-encode the labels, where "p" corresponds to zero-padded areas
    if let=='p':
        return([0,0,0])
    elif let=='b':
        return([1,0,0])
    elif let=='a':
        return([0,1,0])
    elif let=='d':
        return([0,0,1])

def zero_pad(seq):
	if isinstance(seq, str):
		pad = 5000 - (len(seq)-2000)%5000
		seq_ = (pad//2)*'O' + seq + (pad - pad//2)*'O'
	return seq_

#extracting the transcripts for just one chromosome
fasta_seq = SeqIO.parse(open('./data/hg38.fa'), 'fasta')

for fasta in fasta_seq:
	name, sequence = fasta.id, str(fasta.seq)

#file with principal transcripts from GENCODE v33
transcript_file = np.genfromtxt('./data/GENCODE_v33_basic', usecols=(2,3,4,5), dtype='str')

transcripts = []
labels = []

for row in transcript_file:
	# adding the transcripts of the sense strand: whole transcript + flanks + zero-padded, labels + zero-padded
	if row[1]=='+':
		# extract the transcript sequence with 1k flanks
		s = sequence[int(row[2])-1000 : int(row[3])+1000].upper()
		if 'N' not in s:
			# padding labels here 
			pad = 5000 - (len(s)-2000)%5000
			y = (pad//2)*'p' + 'a' + (len(s)-2002)*'b' + 'd' + (pad - pad//2)*'p'
			labels.append(y)
			# padding sequence with Os
			s = zero_pad(s)
			transcripts.append(s)
	# adding the transcripts of the antisense strand
	if row[1]=='-':
		s = sequence[int(row[2])-1000 : int(row[3])+1000].upper()
		if 'N' not in s:
			# padding labels here 
			pad = 5000 - (len(s)-2000)%5000
			y = (pad//2)*'p' + 'a' + (len(s)-2002)*'b' + 'd' + (pad - pad//2)*'p'
			# hot-encoding labels and adding hot-encoded labels to a new list
			labels.append(y)
			# getting complementary seq
			s = ''.join([complementary(x) for x in s])
			# padding sequence with Os
			s = zero_pad(s)
			transcripts.append(s)

print("GENCODE_v33_basic transcripts for hg38: {}".format(len(transcripts)))

#cut these sequences and labels into 5000 chunks
transcripts_ = []
labels_ = []

#transform into chunks 
for i in range(len(transcripts)):
	chunks = (len(transcripts[i])-2000)//5000
	for j in range(1, chunks+1):
		s = transcripts[i]
		l = labels[i]
		transcripts_.append(s[5000*(j-1) : 5000*j+2000])
		labels_.append(l[5000*(j-1) : 5000*j])

np.savetxt('./data/transcripts', transcripts_, fmt='%s', delimiter='\t')
np.savetxt('./data/labels', labels_, fmt='%s', delimiter='\t')
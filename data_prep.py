import numpy as np
from Bio import SeqIO


def complementary(let):
	# A-T, C-G
	if let=='A':
		return 'T'
	if let=='T':
		return 'A'
	if let=='C':
		return 'G'
	if let=='G':
		return 'C'

# genome import, latest version
fasta_seq = SeqIO.parse(open('./data/chr21.fa'), 'fasta')

for fasta in fasta_seq:
	name, sequence = fasta.id, str(fasta.seq)

# file with all principal gene transcripts from GENCODE v33
transcript_file = np.genfromtxt('./data/GENCODE_v33_basic', usecols=(2,3,4,5), dtype='str')

transcripts = []
labels = []

# flanking ends on each side are of this length to include some context
context = 1000

for row in transcript_file:
	# explicitly checking chromosome number
	if row[0]=='chr21':
		s = sequence[int(row[2])-context : int(row[3])+context].upper()
		# adding the transcripts of the sense strand: whole transcript + flanks + zero-padded, labels + zero-padded
		if row[1]=='+':
			# extract the transcript sequence with 1k flanks
			s = sequence[int(row[2])-context : int(row[3])+context].upper()
			if 'N' not in s:
				# padding labels here 
				pad = 5000 - (len(s)-context*2)%5000
				# decrease the pad length from both sides because the context-1 and context+sequence+1 sites are donor and acceptor, respectively
				y = (pad//2-1)*'p' + 'a' + (len(s)-context*2)*'b' + 'd' + (pad - pad//2-1)*'p'
				labels.append(y)
				# padding sequence with Os
				s = (pad//2)*'O' + s + (pad - pad//2)*'O'
				transcripts.append(s)
		# adding the transcripts of the antisense strand
		if row[1]=='-':
			if 'N' not in s:
				# padding labels here 
				pad = 5000 - (len(s)-context*2)%5000
				# decrease the pad length from both sides because the context-1 and context+sequence+1 sites are donor and acceptor, respectively
				y = (pad//2-1)*'p' + 'a' + (len(s)-context*2)*'b' + 'd' + (pad - pad//2-1)*'p'
				# hot-encoding labels and adding hot-encoded labels to a new list
				labels.append(y)
				# getting complementary seq
				s = ''.join([complementary(x) for x in s])
				# padding sequence with Os
				s = (pad//2)*'O' + s + (pad - pad//2)*'O'
				transcripts.append(s)

print("GENCODE_v33_basic transcripts for hg38 chr21: {}".format(len(transcripts)))

# cut these sequences and labels into 5000 chunks
transcripts_chunks = []
labels_chunks = []

# transform into chunks 
for i in range(len(transcripts)):
	chunks = (len(transcripts[i])-context*2)//5000
	for j in range(1, chunks+1):
		s = transcripts[i]
		l = labels[i]
		transcripts_chunks.append(s[5000*(j-1) : 5000*j+context*2])
		labels_chunks.append(l[5000*(j-1) : 5000*j])

np.savetxt('./data/transcripts_chr21', transcripts_chunks, fmt='%s', delimiter='\t')
np.savetxt('./data/labels_chr21', labels_chunks, fmt='%s', delimiter='\t')
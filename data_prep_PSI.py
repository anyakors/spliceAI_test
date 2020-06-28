from Bio import SeqIO

from utils import *

# genome import, latest version
fasta_seq = SeqIO.parse(open('./data/chr21.fa'), 'fasta')

for fasta in fasta_seq:
    name, sequence = fasta.id, str(fasta.seq)

# file with all principal gene transcripts from GENCODE v33
hexevent = np.genfromtxt('./data/HEXevent_chr21.txt', dtype='str', comments=None, skip_header=1)

exons = form_transcripts(hexevent)

# flanking ends on each side are of this length to include some context
context = 1000

transcripts = []
labels = []

for row in zip(exons['strand'], exons['exons'], exons['incl']):
    s = sequence[int(row[1][0]) - context: int(row[1][-1]) + context].upper()
    pad = 5000 - (len(s) - context * 2) % 5000

    if row[0] == '+' and 'N' not in s and len(s)-2*context>300:

        es, ee = [int(x) for x in row[1][0::2]], [int(x) for x in row[1][1::2]]
        es, ee = [(i - es[0]) for i in es], [(i - es[0]) for i in ee]

        y = [0]*(len(s) - context*2)

        for x in zip(es, ee, row[2]):
            y[x[0]:x[1]] = [float(x[2])]*(x[1]-x[0])

        y = [-1]*(pad // 2) + y + [-1]*(pad - pad // 2)
        labels.append(y)

        s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'
        transcripts.append(s)

    if row[0] == '-' and 'N' not in s and len(s)-2*context>300:

        es, ee = [int(x) for x in row[1][0::2]], [int(x) for x in row[1][1::2]]
        es, ee = [(i - es[0]) for i in es], [(i - es[0]) for i in ee]

        y = [0]*(len(s) - context*2)

        for x in zip(es, ee, row[2]):
            y[x[0]:x[1]] = [float(x[2])]*(x[1]-x[0])

        y = [-1]*(pad // 2) + y + [-1]*(pad - pad // 2)
        labels.append(y)

        s = ''.join([complementary(x) for x in s])
        s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'
        transcripts.append(s)


# cut these sequences and labels into 5000 chunks
transcripts_chunks = []
labels_chunks = []

# transform into chunks 
for i in range(len(transcripts)):
    chunks = (len(transcripts[i]) - context * 2) // 5000
    for j in range(1, chunks + 1):
        s = transcripts[i]
        l = labels[i]
        transcripts_chunks.append(s[5000 * (j - 1): 5000 * j + context * 2])
        labels_chunks.append(l[5000 * (j - 1): 5000 * j])

np.savetxt('./data/transcripts_HEX_chr21', transcripts_chunks, fmt='%s', delimiter='\t')
np.savetxt('./data/labels_HEX_chr21', labels_chunks, fmt='%s', delimiter='\t')

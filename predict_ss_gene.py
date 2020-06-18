import numpy as np
from Bio import SeqIO

import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go

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

def hot_encode_seq(let):
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
    if let=='p':
        return([0,0,0])
    elif let=='b':
        return([1,0,0])
    elif let=='a':
        return([0,1,0])
    elif let=='d':
        return([0,0,1])

def dehot_encode_pred(let):
    if np.argmax(let)==0:
        return('b')
    elif np.argmax(let)==1:
        return('a')
    elif np.argmax(let)==2:
        return('d')

def dehot_encode_label(let):
    if (let==[0,0,0]).all():
        return('p')
    elif (let==[1,0,0]).all():
        return('b')
    elif (let==[0,1,0]).all():
        return('a')
    elif (let==[0,0,1]).all():
        return('d')

def make_labels(s, context, es, ee):

	es, ee = [int(i)-int(es[0]) for i in es], [int(i)-int(es[0]) for i in ee]
	y = 'b'*(len(s)-context*2+2)

	for i in range(len(es)):
		y = y[:es[i]]+'a'+y[es[i]+1:ee[i]]+'d'+y[ee[i]+1:]

	pad = 5000 - (len(s)-context*2)%5000
	y = (pad//2-1)*'p' + y + (pad - pad//2-1)*'p'

	return y

def label_to_exons(y, pad):
    
    y1 = y[pad//2-1:-pad//2+1]
    es1 = [pos for pos, char in enumerate(y1) if char == 'a']
    ee1 = [pos-1 for pos, char in enumerate(y1) if char == 'd']
    
    return es1, ee1

def transform_input(transcripts_, labels_):

    transcripts = []
    labels = []
    # hot-encode
    for i in range(len(transcripts_)):
        # hot-encode seq
        transcripts.append([np.array(hot_encode_seq(let)) for let in transcripts_[i]])
        # hot-encode labels
        labels.append([np.array(hot_encode_label(x)) for x in labels_[i]])

    return transcripts, labels

def transform_output(y_test, y_pred):
    y_test_, y_pred_ = [], []
    for vector in y_test:
        y_test_.append([dehot_encode_label(x) for x in vector])
    for vector in y_pred:
        y_pred_.append([dehot_encode_pred(x) for x in vector])
    return y_test_, y_pred_

# genome import, latest version
fasta_seq = SeqIO.parse(open('./data/chr21.fa'), 'fasta')

for fasta in fasta_seq:
	name, sequence = fasta.id, str(fasta.seq)

# file with all principal gene transcripts from GENCODE v33
transcript_file = np.genfromtxt('./data/GENCODE_v33_basic', usecols=(1,3,4,5,9,10), dtype='str')

transcript_name = 'ENST00000612267'

# flanking ends on each side are of this length to include some context
context = 1000

for row in transcript_file:
	# explicitly checking transcript_name
	if transcript_name in row[0]:
		# sequence from start to end
		s = sequence[int(row[2])-context : int(row[3])+context].upper()
		# adding the transcripts of the sense strand: whole transcript + flanks + zero-padded, labels + zero-padded
		if row[1]=='+':
			# extract the transcript sequence with 1k flanks
			if 'N' not in s:
				# padding labels here 
				pad = 5000 - (len(s)-context*2)%5000
				es, ee = row[4].split(',')[:-1], row[5].split(',')[:-1]
				# decrease the pad length from both sides because the context-1 and context+sequence+1 sites are donor and acceptor, respectively
				y = make_labels(s, context, es, ee)
				# padding sequence with Os
				s = (pad//2)*'O' + s + (pad - pad//2)*'O'
		# adding the transcripts of the antisense strand
		if row[1]=='-':
			if 'N' not in s:
				# padding labels here 
				pad = 5000 - (len(s)-context*2)%5000
				# decrease the pad length from both sides because the context-1 and context+sequence+1 sites are donor and acceptor, respectively
				es, ee = row[4].split(',')[:-1], row[5].split(',')[:-1]
				# decrease the pad length from both sides because the context-1 and context+sequence+1 sites are donor and acceptor, respectively
				y = make_labels(s, context, es, ee)
				# hot-encoding labels and adding hot-encoded labels to a new list
				# getting complementary seq
				s = ''.join([complementary(x) for x in s])
				# padding sequence with Os
				s = (pad//2)*'O' + s + (pad - pad//2)*'O'
		break

# cut these sequences and labels into 5000 chunks
transcript_chunks = []
label_chunks = []

# transform into chunks 
chunks = (len(s)-context*2)//5000
for j in range(1, chunks+1):
	transcript_chunks.append(s[5000*(j-1) : 5000*j+context*2])
	label_chunks.append(y[5000*(j-1) : 5000*j])

# PREDICT -> y_pred

x_test, y_test = transform_input(transcript_chunks, label_chunks)

x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.load_model('./data/model_spliceAI2k')

y_pred = model.predict(x_test)

# Quantify

y_test_, y_pred_ = transform_output(y_test, y_pred)

donor_t_p, acceptor_t_p, blank_t_p = 0, 0, 0
donor, acceptor, blank = 0, 0, 0 

for i in range(len(y_test)):
    for j in range(len(y_test[0])):
        if y_test[i][j]==y_pred[i][j] and y_test[i][j]=='d':
            donor += 1
            donor_t_p += 1
        elif y_test[i][j]==y_pred[i][j] and y_test[i][j]=='a':
            acceptor += 1
            acceptor_t_p += 1
        elif y_test[i][j]==y_pred[i][j] and y_test[i][j]=='b':
            blank += 1
            blank_t_p += 1
        elif y_test[i][j]=='d':
            donor += 1
        elif y_test[i][j]=='a':
            acceptor += 1
        elif y_test[i][j]=='b':
            blank += 1

# Plot

es_pred, ee_pred = label_to_exons(y_pred, pad)
es, ee = [int(i)-int(es[0]) for i in es], [int(i)-int(es[0]) for i in ee]

def add_exon_real(x_start, x_end):
	exon = go.Scatter(
					x=[x_start, x_end],
					y=[0.6,0.6],
					mode='lines+markers',
					marker = dict(
						color = 'rgb(55, 255, 55)',
						size = 6,
					),
					line=dict(color='rgb(55, 255, 55)', width=2),
					)
	return exon

def add_exon_pred(x_start, x_end):
	exon = go.Scatter(
					x=[x_start, x_end],
					y=[0.5,0.5],
					mode='markers',
					marker = dict(
						color = 'rgb(255, 55, 55)',
						size = 6,
					),
					)
	return exon

data = []

for x in zip(es, ee):
	data.append(add_exon_real(x[0], x[1]))

for x in zip(es_pred, ee_pred):
	data.append(add_exon_pred(x[0], x[1]))

layout = go.Layout(title='junctions AF254983.1')

fig = go.Figure(data=data, layout=layout)
fig['layout']['yaxis'].update(title='', range=[0.0, 1.0])
pyo.plot(fig, filename='junctions_lines.html')
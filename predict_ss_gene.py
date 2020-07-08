from utils import *
from model import topk_accuracy_

import numpy as np
from Bio import SeqIO

import tensorflow as tf

import plotly.offline as pyo
import plotly.graph_objs as go

# genome import, latest version
fasta_seq = SeqIO.parse(open('./data/chr21.fa'), 'fasta')

for fasta in fasta_seq:
    name, sequence = fasta.id, str(fasta.seq)

# file with all principal gene transcripts from GENCODE v33
transcript_file = np.genfromtxt('./data/GENCODE_v33_basic', usecols=(1, 3, 4, 5, 9, 10, 12), dtype='str')
canonical = np.genfromtxt('./data/GENCODE_v32_hg38_canonical_chr21', usecols=(4,), dtype='str')

gene_name = 'TIAM1' # AF254983.1

# flanking ends on each side are of this length to include some context
context = 1000

for row in transcript_file:
    # explicitly checking transcript_name
    if row[6]==gene_name and row[0] in canonical:
        print(row[6], row[0])
        # sequence from start to end
        s = sequence[int(row[2]) - context: int(row[3]) + context].upper()
        # adding the transcripts of the sense strand: whole transcript + flanks + zero-padded, labels + zero-padded
        if row[1] == '+':
            # extract the transcript sequence with 1k flanks
            if 'N' not in s:
                # padding labels here
                pad = 5000 - (len(s) - context * 2) % 5000
                es, ee = row[4].split(',')[:-1], row[5].split(',')[:-1]
                # decrease the pad length from both sides because the context-1 and context+sequence+1 sites are
                # donor and acceptor, respectively
                y = make_labels(s, context, es, ee)
                # padding sequence with Os
                s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'
        # adding the transcripts of the antisense strand
        if row[1] == '-':
            if 'N' not in s:
                # padding labels here
                pad = 5000 - (len(s) - context * 2) % 5000
                # decrease the pad length from both sides because the context-1 and context+sequence+1 sites are
                # donor and acceptor, respectively
                es, ee = row[4].split(',')[:-1], row[5].split(',')[:-1]
                # decrease the pad length from both sides because the context-1 and context+sequence+1 sites are
				# donor and acceptor, respectively
                y = make_labels(s, context, es, ee)
                # hot-encoding labels and adding hot-encoded labels to a new list
                # getting complementary seq
                s = ''.join([complementary(x) for x in s])
                # padding sequence with Os
                s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'
        break

# cut these sequences and labels into 5000 chunks
transcript_chunks = []
label_chunks = []

# transform into chunks 
chunks = (len(s) - context * 2) // 5000
for j in range(1, chunks + 1):
    transcript_chunks.append(s[5000 * (j - 1): 5000 * j + context * 2])
    label_chunks.append(y[5000 * (j - 1): 5000 * j])

# PREDICT -> y_pred

x_test, y_test = transform_input(transcript_chunks, label_chunks)

x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.load_model('./data/model_spliceAI2k_chr1_3', compile=False)

y_pred = model.predict(x_test)
acc = topk_accuracy_(y_test, y_pred)
print('Top-k accuracy: {:.2f}'.format(acc))

# Extract topk

y_test = y_test.reshape([len(y_test) * 5000, 3])
y_pred = y_pred.reshape([len(y_pred) * 5000, 3])

a_true, d_true = np.nonzero(y_test[:, 1]), np.nonzero(y_test[:, 2])
k = len(a_true[0])

a_pred, d_pred = y_pred[:, 1], y_pred[:, 2]
a_pred_topk = np.argsort(a_pred, axis=-1)[-k:]
d_pred_topk = np.argsort(d_pred, axis=-1)[-k:]

# Plot

def add_exon_real(x_start, x_end):
    exon = go.Scatter(
        x=[x_start, x_end],
        y=[0.6, 0.6],
        mode='lines+markers',
        marker=dict(
            color='rgb(55, 255, 55)',
            size=6,
        ),
        line=dict(color='rgb(55, 255, 55)', width=2),
    )
    return exon


def add_exon_pred(x_start):
    exon = go.Scatter(
        x=[x_start],
        y=[0.5],
        mode='markers',
        marker=dict(
            color='rgb(255, 55, 55)',
            size=6,
        ),
    )
    return exon


def add_true_line(x):
    line = go.Scatter(
        x=[x, x],
        y=[0.5, 0.6],
        mode='lines',
        line=dict(color='rgb(55, 255, 55)', width=2),
    )
    return line

data = []

for x in zip(a_true[0], d_true[0]):
    data.append(add_exon_real(x[0], x[1]))

for x in a_pred_topk:
    data.append(add_exon_pred(x))

for x in d_pred_topk:
    data.append(add_exon_pred(x))

for x in np.intersect1d(a_true, a_pred_topk):
    data.append(add_true_line(x))

for x in np.intersect1d(d_true, d_pred_topk):
    data.append(add_true_line(x))

layout = go.Layout(title='junctions TIAM1, topk = {:.2f}'.format(acc))

fig = go.Figure(data=data, layout=layout)
fig['layout']['yaxis'].update(title='', range=[0.0, 1.0])
pyo.plot(fig, filename='junctions_lines_TIAM1.html')

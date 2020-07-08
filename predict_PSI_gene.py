from utils import *
from model import topk_accuracy_

import numpy as np
from Bio import SeqIO

import tensorflow as tf

import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.express as px

# genome import, latest version
fasta_seq = SeqIO.parse(open('./data/chr21.fa'), 'fasta')

for fasta in fasta_seq:
    name, sequence = fasta.id, str(fasta.seq)

# file with all principal gene transcripts from GENCODE v33
transcript_file = np.genfromtxt('./data/GENCODE_v33_basic', usecols=(1, 3, 4, 5, 9, 10, 12), dtype='str')
hexevent = np.genfromtxt('./data/HEXevent_chr21.txt', dtype='str', comments=None, skip_header=1)

gene_name = 'TIAM1' # AF254983.1
exons = form_transcript(hexevent, gene_name)

print(exons)

# flanking ends on each side are of this length to include some context
context = 1000

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

        s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'

    if row[0] == '-' and 'N' not in s and len(s)-2*context>300:

        es, ee = [int(x) for x in row[1][0::2]], [int(x) for x in row[1][1::2]]
        es, ee = [(i - es[0]) for i in es], [(i - es[0]) for i in ee]

        y = [0]*(len(s) - context*2)

        for x in zip(es, ee, row[2]):
            y[x[0]:x[1]] = [float(x[2])]*(x[1]-x[0])

        y = [-1]*(pad // 2) + y + [-1]*(pad - pad // 2)

        s = ''.join([complementary(x) for x in s])
        s = (pad // 2) * 'O' + s + (pad - pad // 2) * 'O'


# cut these sequences and labels into 5000 chunks
transcript_chunks = []
labels_chunks = []

# transform into chunks 
chunks = (len(s) - context * 2) // 5000
for j in range(1, chunks + 1):
    transcript_chunks.append(s[5000 * (j - 1): 5000 * j + context * 2])
    labels_chunks.append(y[5000 * (j - 1): 5000 * j])

# PREDICT -> y_pred

transcript_chunks_ = []
labels_chunks_ = []

for i in range(len(transcript_chunks)):
    # hot-encode seq
    transcript_chunks_.append([np.array(hot_encode_seq(let)) for let in transcript_chunks[i]])
    labels_chunks_.append([float(x) for x in labels_chunks[i]])

x_test = np.array(transcript_chunks_)
y_test = np.array(labels_chunks_)

model = tf.keras.models.load_model('./data/model_regression_HEX_chr21', compile=False)

y_pred = model.predict(x_test)

# Extract topk

y_test = y_test.reshape(len(y_test) * 5000)
y_pred = y_pred.reshape(len(y_pred) * 5000)

# Plot

data = []

data.append(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', line=dict(color='rgb(55, 255, 55)', width=5)))
data.append(go.Scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', line=dict(color='rgb(255, 55, 55)', width=5)))

layout = go.Layout(title='PSI TIAM1')

fig = go.Figure(data=data, layout=layout)
fig['layout']['yaxis'].update(title='', range=[-0.5, 1.8])
pyo.plot(fig, filename='PSI_TIAM1.html')

#fig = px.line(x=np.arange(len(y_test)), y=y_test)
#fig.show()
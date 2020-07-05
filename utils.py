import numpy as np


def complementary(let):
    # A-T, C-G
    if let == 'A':
        return 'T'
    if let == 'T':
        return 'A'
    if let == 'C':
        return 'G'
    if let == 'G':
        return 'C'


def hot_encode_seq(let):
    if let == 'A':
        return ([1, 0, 0, 0])
    elif let == 'T':
        return ([0, 1, 0, 0])
    elif let == 'C':
        return ([0, 0, 1, 0])
    elif let == 'G':
        return ([0, 0, 0, 1])
    elif let == 'O':
        return ([0, 0, 0, 0])


def hot_encode_label(let):
    if let == 'p':
        return ([0, 0, 0])
    elif let == 'b':
        return ([1, 0, 0])
    elif let == 'a':
        return ([0, 1, 0])
    elif let == 'd':
        return ([0, 0, 1])


def dehot_encode_pred(let):
    if np.argmax(let) == 0:
        return ('b')
    elif np.argmax(let) == 1:
        return ('a')
    elif np.argmax(let) == 2:
        return ('d')


def dehot_encode_label(let):
    if (let == [0, 0, 0]).all():
        return ('p')
    elif (let == [1, 0, 0]).all():
        return ('b')
    elif (let == [0, 1, 0]).all():
        return ('a')
    elif (let == [0, 0, 1]).all():
        return ('d')


def make_labels(s, context, es, ee):
    es, ee = [int(i) - int(es[0]) for i in es], [int(i) - int(es[0]) for i in ee]
    y = 'b' * (len(s) - context * 2 + 2)

    for i in range(len(es)):
        y = y[:es[i]] + 'a' + y[es[i] + 1:ee[i]] + 'd' + y[ee[i] + 1:]

    pad = 5000 - (len(s) - context * 2) % 5000
    y = (pad // 2 - 1) * 'p' + y + (pad - pad // 2 - 1) * 'p'

    return y


def label_to_exons(y, pad):
    y_ = []
    for row in y:
        y_.extend(row)
    y1 = y_[pad // 2 - 1:-pad // 2 + 1]
    es1 = [pos for pos, char in enumerate(y1) if char == 'a']
    ee1 = [pos - 1 for pos, char in enumerate(y1) if char == 'd']

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


def form_transcripts(hexevent):
    transcripts = {'gene': [], 'strand': [], 'exons': [], 'incl': []}
    c = None
    l = []
    n = []
    for row in hexevent:
        if row[-1]!='onlyEST':
            if c is None:
                c = row[-1]
                l.extend([row[2], row[3]])
                n.extend([row[10]])
            elif c is not None and c==row[-1]:
                l.extend([row[2], row[3]])
                n.extend([row[10]])
            else:
                transcripts['gene'].append(c)
                transcripts['strand'].append(row[1])
                transcripts['exons'].append(l)
                transcripts['incl'].append(n)
                c = row[-1]
                l = [row[2], row[3]]
                n = [row[10]]
    return transcripts


def form_transcript(hexevent, gene_name):
    transcript = {'gene': [], 'strand': [], 'exons': [], 'incl': []}
    c = None
    l = []
    n = []
    for row in hexevent:
        if row[-1]==gene_name:
            l.extend([row[2], row[3]])
            n.extend([row[10]])
            if c==None:
                transcript['gene'].append(gene_name)
                transcript['strand'].append(row[1])
                c = 1

    transcript['exons'].append(l)
    transcript['incl'].append(n)

    return transcript
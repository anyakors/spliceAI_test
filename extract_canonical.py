import numpy as np
from Bio import SeqIO

transcript_file = np.genfromtxt('./data/GENCODE_v24lift37_hg19', usecols=(1, 4, 5, 12), dtype='str')

print("GENCODE v24lift37 annotations length for hg19:", len(transcript_file))

canonical = []

for i in range(len(transcript_file)):
    # row represents one transcript
    row = transcript_file[i]
    if row[3] not in [x[0] for x in canonical]:
        # taking the first as longest for now
        longest = {'l': int(row[2])-int(row[1]), 't': (row[0].split('.'))[0]}
        for j in range(i+1, len(transcript_file)):
            if transcript_file[j][3]==row[3]:
                #compare lengths
                if int(transcript_file[j][2])-int(transcript_file[j][1])>longest['l']:
                    longest['l'] = int(transcript_file[j][2])-int(transcript_file[j][1])
                    longest['t'] = (transcript_file[j][0].split('.'))[0]
            else:
                canonical.append([row[3], longest['t']])
                break

print("GENCODE v24lift37 longest annotations for each gene:", len(canonical))

np.savetxt('./data/transcripts_canonical_1', canonical, fmt='%s', delimiter='\t')

canonical_1 = np.genfromtxt('./data/canonical_dataset.txt', usecols=(0), dtype='str')

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

not_in_their = intersection([x[0] for x in canonical], canonical_1)
print("Some genes not in their dataset:", len(not_in_their))

not_in_mine = intersection(canonical_1, [x[0] for x in canonical])
print("Some genes not in my dataset:", len(not_in_mine))
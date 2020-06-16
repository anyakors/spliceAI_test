# spliceAI_test

This is a test repository with code to reproduce Jaganathan et al. "Predicting Splicing from Primary Sequence with Deep Learning" (2019) Cell paper results on splicing prediction with deep learning. The architecture reproduced is SpliceAI-2k with 1000 nt flanks on both ends of the sequence: 3 stacks of 4 RB blocks with 3 Conv shortcuts. The data is flattened, so all the layers are in 1D.

The sequence files for training are too heavy for a github repo, so the data_prep.py script is included. To use it, download hg38.fa file of the latest human genome assembly into the ./data folder:

```
cd ./data
wget --timestamping 'ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz' -O hg38.fa.gz
gunzip hg38.fa.gz
```

and run the data_prep.py script. It will generate files with transcript sequences using GENCODE_v33_basic transcript annotations (downloaded from UCSC genome browser, Basic Gene Annotation Set from GENCODE Version 33 (Ensembl 99)) in chunks of 7000 nt length (with 1000nt context on each side) for the sequences and 5000 nt length for the labels with zero-pads as follows: 

![alt text](https://github.com/iamqoqao/spliceAI_test/blob/master/labels.png?raw=true)

where 'p' is the pad indication in the labels and 'O' is the pad indication for the seq, and acceptor and donor sites are the last/first sites of the context sequence (transcript sequence starts with the exon, but acceptor and donor sites are the last and the first sites of the intron, so they're adjacent to the exon on the left/right). Then the training script might be run.

# spliceAI_test

This is a test repository with code to reproduce Jaganathan et al. "Predicting Splicing from Primary Sequence with Deep Learning" (2019) Cell paper results on splicing prediction with deep learning. The architecture reproduced is SpliceAI-2k with 1000 nt flanks on both ends of the sequence: 3 stacks of 4 RB blocks with 3 Conv shortcuts. The data is flattened, so all the layers are in 1D.

The transcript files for training are too heavy for a github repo, so the script extract_transcript.py is included. In order to use it, one must put hg38.fa file of hg38 to the ./data folder and run the script. It will generate files with transcript sequences in chunks of 7000 nt length for the sequences and 5000 nt length for the labels. Then the training script might be used.

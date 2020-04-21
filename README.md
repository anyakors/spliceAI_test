# spliceAI_test

This is a test repository with code to reproduce Jaganathan et al. (2019) Cell paper results on splicing prediction. 

The transcript files for training are too heavy for a github repo, so the script extract_transcript.py is included. In order to use it, one must put chr20.fa file of hg38 to the ./data folder and run the script. It will generate files with transcript sequences in chunks of 7000 nt length for the sequences and 5000 nt length for the labels (2000 nt is the sum of flanks on both sides). Then the training script might be used.

# Language Classification
All the codes for this project should be placed at the upper directory of train folder. The codes are divided into 3 parts:
- Pre-processing (PreProcessing.py)
- RNN training (RNN_train.py)
- RNN streaming (RNN_stream.py)
  
**`PreProcessing.py`**:

This takes each of the audio files at a time and first removes the silence from each file. For removing the silence, `librosa.effects.split` was used to find the non-silent intervals and then only these intervals were considered to generate the features for. Those intervals that were not returned, were removed. Then, for each non-silent 1-D array of data loaded, `librosa.feature.mfcc` was used to generate the MFCC features of it and splits it into sequences of length, which was set to 500 here. Simply put, after combining all audio files for each class, there will be `(N_seq_class, N_seqLength, N_features) = (N_seq_class, 500, 64)`. Then it generates class labels of shape `(N_seq_class, 500, 3)` for each class. The labels are set to be 1-hot arrays `[1, 0, 0]`, `[0, 1, 0]` and `[0, 0, 1]` for English, Hindi and Mandarin, respectively.
Then all these classes data are concatenated and shuffled to randomize the order of sequences. To concatenate the data, we trimmed the portion of data of each class that remained after dividing the number of total instants by 500. The last 10% of the combined data is then used as the unseen test data that was used in RNN streaming. From the rest 90%, the train-validation ratio was set to 80%-20%. After doing this, the data are saved as arrays of X_train, y_train, X_test, and y_test in `Data.hd5 file`.

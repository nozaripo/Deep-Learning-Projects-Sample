import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model
import librosa, librosa.display
import os
from sklearn.utils import shuffle

#### generate toy data
root_dir = os.getcwd()
eng_dir = root_dir + '/train/train_english'
hin_dir = root_dir + '/train/train_hindi'
man_dir = root_dir + '/train/train_mandarin'

# define the list of languages audio files directories
lang_list = list([eng_dir, hin_dir, man_dir])

# arrays used to stack up the sequqnces for each language
x_eng = np.empty(shape=[0, 64])
x_hin = np.empty(shape=[0, 64])
x_man = np.empty(shape=[0, 64])

ii = 0
for dir_name in lang_list:
    os.chdir(dir_name)
    ii+=1
    print(ii)    
    for filename in os.listdir(os.getcwd()):
        x, fs = librosa.load(filename, sr=16000)
        inter_50 = librosa.effects.split(x, top_db=50) #audio above 20dB

        x_50 = np.empty(shape=[1])

        for i in inter_50:
            start,end = i
            x_50 = np.concatenate((x_50, x[start:end]), axis=0)

        # librosa.display.waveplot(x, sr=fs)
        mfccs = librosa.feature.mfcc(x_50, sr=fs, n_mfcc=64,
                                    n_fft=int(fs*.025),
                                    hop_length=int(fs*.01))
        if dir_name == lang_list[0]:
            x_eng = np.concatenate((x_eng, mfccs.T), axis=0)
            # print('eng shape+= ',x_eng.shape)
        elif dir_name == lang_list[1]:
            x_hin = np.concatenate((x_hin, mfccs.T), axis=0)
            # print('hin shape+= ',x_hin.shape) 
        else:
            x_man = np.concatenate((x_man, mfccs.T), axis=0)
            # print('man shape+= ',x_man.shape) 


# find a portion of each class divisible by sequence length S=500 and trim the rest
S = 500
M_eng = x_eng.shape[0]
M_hin = x_hin.shape[0]
M_man = x_man.shape[0]

eng_trim_idx = np.arange(np.mod(M_eng,S))
hin_trim_idx = np.arange(np.mod(M_hin,S))
man_trim_idx = np.arange(np.mod(M_man,S))

x_eng = np.delete(x_eng, eng_trim_idx, 0)
x_hin = np.delete(x_hin, hin_trim_idx, 0)
x_man = np.delete(x_man, man_trim_idx, 0)

print(x_eng.shape)

# create labels 
M_eng = x_eng.shape[0]
M_hin = x_hin.shape[0]
M_man = x_man.shape[0]

y_eng = np.ones((M_eng,1))@np.array([[1, 0, 0]])
y_hin = np.ones((M_hin,1))@np.array([[0, 1, 0]])
y_man = np.ones((M_man,1))@np.array([[0, 0, 1]])


# reshape the features and labels, e.g., (M_eng,64) to (N_eng, S, 64)
N_eng = int(M_eng/S)
N_hin = int(M_hin/S)
N_man = int(M_man/S)

X_eng = x_eng.reshape(N_eng, S, 64)
X_hin = x_hin.reshape(N_hin, S, 64)
X_man = x_man.reshape(N_man, S, 64)
Y_eng = y_eng.reshape(N_eng, S, 3)
Y_hin = y_hin.reshape(N_hin, S, 3)
Y_man = y_man.reshape(N_man, S, 3)


# Concatenate all X's and y's separately
X = np.concatenate((X_eng, X_hin, X_man), axis=0)
y = np.concatenate((Y_eng, Y_hin, Y_man), axis=0)

print(X.shape)
print(y.shape)

# Concatenate everything and shuffle
data_concat = np.concatenate((X,y), axis=2)
data_concat_shuff = shuffle(data_concat)

print(data_concat_shuff.shape)

# set the size of the train and test data: test set is 10% of the data
# the train data will include the validation set as well
# test set will be used as unseen portion of data for streaming prediction
train_len = int(np.ceil(len(y)*.9))
test_len =  len(y)-train_len


# split training and test data
train_data = data_concat_shuff[:train_len,]
test_data  = data_concat_shuff[-test_len:,]


X_train = train_data[:,:,:64]
y_train = train_data[:,:,-3:]
X_test  = test_data[:,:,:64]
y_test  = test_data[:,:,-3:]


# print(S)
# print(X_eng.shape)
# print(X_hin.shape)
# print(X_man.shape)
# print(Y_eng.shape)
# print(Y_hin.shape)
# print(Y_man.shape)

# print(Y_eng[:5,:1,])

# Go to the root directory where the code is
os.chdir(root_dir)   


# create an hdf5 file and save the inputs and outputs
hf = h5py.File('Data.hd5', 'w')


hf.create_dataset('X_train', data=X_train)
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('y_test', data=y_test)

hf.close()






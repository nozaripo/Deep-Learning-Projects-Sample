import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



# Demo flag: set to True to see the probability prediction vs the true labels for each instant in sequences
Demo_flag = True

# Pred_acc flag: set to True to compute the accuracy of the streaming model on the unseen test data
Pred_acc_flag = True

data = h5py.File('Data.hd5','r')

X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test  = np.array(data['X_test'])
y_test  = np.array(data['y_test'])

test_len = len(y_test)


##### define the streaming-infernece model
streaming_in = Input(batch_shape=(1, None, 64))  ## stateful ==> needs batch_shape specified
foo = GRU(200, return_sequences=True, stateful=True )(streaming_in)
foo = Dense(100, activation='relu')(foo)
streaming_pred = Dense(3, activation='softmax')(foo)
streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

streaming_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
streaming_model.summary()


#this will produce a digram of the model 
plot_model(streaming_model, to_file='streaming_model_arch.png', show_shapes=True, show_layer_names=True)

##### copy the weights from trained model to streaming-inference model

streaming_model.load_weights('weights.hdf5')

streaming_model.save('Stream_RNN.hdf5')




##### demo the behaivor
if Demo_flag:
    print('Demo Mode:')
    # for s in range(test_len):
    for s in range(2):
        for n in range(2):
            true_label = y_test[s][n].reshape(1,1,3)

            single_input = X_test[s][n].reshape(1,1,64)
            stream_single_prob = streaming_model.predict(single_input)
            print(f'@ sequence {s}, instant {n}:\n stream_probability= {stream_single_prob}  //  true_label= {true_label}')

        streaming_model.reset_states()



if Pred_acc_flag:
    stream_pred_correct = 0
    print('Predict Mode:')
    for s in range(test_len):
        
        pred_compare = np.zeros((1,500))
        true_label = y_test[s].reshape(1,500,3)
        single_input = X_test[s].reshape(1,500,64)
        stream_single_prob = streaming_model.predict(single_input)

        stream_pred_hot = np.argmax(stream_single_prob,axis=2)
        true_label_hot = np.argmax(true_label,axis=2)
        # single_pred = streaming_model.predict(in_feature_vector)[0][0]
        pred_compare[stream_pred_hot==true_label_hot]=1
        
        stream_pred_correct += np.sum(pred_compare)
        # print(f'Seq-model Prediction, Streaming-Model Prediction, difference [{n}]: {seq_pred[n] : 3.2f}, {single_pred : 3.2f}, {seq_pred[n] - single_pred: 3.2f}')
        streaming_model.reset_states()

        if np.mod(s,100)==0:
            print(f'{round(100*stream_pred_correct/((s+1)*500),2)}% prediction accuracy for {s} sequences')

    stream_pred_acc = stream_pred_correct/(test_len*500)
    print(f'\nThe streaming model prediction accuracy on the entire test set is:  {round(100*stream_pred_acc,2)}%')




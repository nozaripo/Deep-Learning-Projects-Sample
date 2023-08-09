import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



# generate class weights
def create_weights_matrix(labels): 
    
    weights_mapping = {0:1, 1:3, 2:2}
    
    # If using "return_sequences = True"
    if labels.ndim == 3:
        weights_matrix = np.zeros(labels.shape[0:2])

        for i,sample in enumerate(labels):
            for j,elem in enumerate(sample):
                weights_matrix[i,j] = weights_mapping[elem[0]]
    
    else:
        weights_matrix = np.zeros(labels.shape[0])
        for i,sample in enumerate(labels):
            weights_matrix[i] = weights_mapping[sample]
    
    return weights_matrix


data = h5py.File('Data.hd5','r')

X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test  = np.array(data['X_test'])
y_test  = np.array(data['y_test'])


##################################################
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_test[:,1,:])

train_len = len(y_train)
print(train_len)
# Compute 1-hot labels and then class weights
hot_labels = np.argmax(y_train,axis=2).reshape(train_len,500,1)
class_weights = create_weights_matrix(hot_labels)



# print(y_test_labels.shape)
# print(labels[:,1,:])


######  Define/Build/Train Training Model
training_in_shape = X_train.shape[1:]
training_in = Input(shape=training_in_shape)
print(training_in)
# training_in = Input(batch_shape=(None,train_seq_length,feature_dim)) this works too
model = GRU(200, return_sequences=True, stateful=False)(training_in)
model = Dense(100, activation='relu')(model)
training_pred = Dense(3, activation='softmax')(model)

training_model = Model(inputs=training_in, outputs=training_pred)
training_model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics= ['accuracy'], sample_weight_mode='temporal')
training_model.summary()

#this will produce a digram of the model 
plot_model(training_model, to_file='training_model_arch.png', show_shapes=True, show_layer_names=True)

# fit the model to the data
results = training_model.fit(X_train, y_train, batch_size=128, epochs=100, 
                            validation_split=0.2, shuffle = True, sample_weight=class_weights)


# plot our learning curves
loss = results.history['loss']
val_loss = results.history['val_loss']
acc = results.history['accuracy']
val_acc = results.history['val_accuracy']
epochs = np.arange(len(loss))+1

plt.figure(1)
plt.plot(epochs, loss, label='Training')
plt.plot(epochs, val_loss, label='Validation')
plt.xlabel('Epochs #')
plt.ylabel('Multiclass Cross Entropy Loss')
plt.title('Loss vs. epochs: GRU with 500 units')
plt.legend()
plt.savefig('Loss.png', dpi=256)

plt.figure(2)
plt.plot(epochs, acc, label='Training')
plt.plot(epochs, val_acc, label='Validation')
plt.xlabel('Epochs #')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. epochs: GRU with 500 units')
plt.legend()
plt.savefig('Accuracy.png', dpi=256)



training_model.save_weights('weights.hdf5', overwrite=True)


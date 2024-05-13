#!/usr/bin/env python3
"""
Define and train the neural network
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Network and training parameters
NB_CLASSES = 24
BATCH_SIZE = 128
EPOCHS = 200
VALIDATION_SPLIT = 0.2
N_HIDDEN = 128
DROPOUT = 0.3

# Load the data
file_path = '../dataset/asl_train_preprocessed.csv'
df_asl_train = pd.read_csv(file_path)
file_path = '../dataset/asl_test_preprocessed.csv'
df_asl_test = pd.read_csv(file_path)

# Get train and test data and labels (X-train, X_test, Y_train, Y_test)
X_train_data = df_asl_train.iloc[:, 1:]
X_test_data = df_asl_test.iloc[:, 1:]
X_train = np.array(X_train_data.values).reshape(X_train_data.shape[0], 784)
X_test = np.array(X_test_data.values).reshape(X_test_data.shape[0], 784)

Y_train_data = df_asl_train.iloc[:,0]
Y_test_data = df_asl_test.iloc[:,0]
Y_train = np.array(Y_train_data.values).reshape(Y_train_data.shape[0])
Y_test= np.array(Y_test_data.values).reshape(Y_test_data.shape[0])

# Normalize inputs to be within in [0, 1]
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255

# One hot representation of the labels
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=NB_CLASSES + 1, dtype='int64')
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=NB_CLASSES + 1, dtype='int64')

# Build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(784,),
                             name='dense_layer',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
                             name='dense_layer_2',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES + 1,
                             name='dense_layer_3',
                             activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer='RMSProp',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=True,
          validation_split=VALIDATION_SPLIT,
          shuffle=False)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

plt.plot(history.history['accuracy'],
         label='accuracy',
         color='red')
plt.plot(history.history['val_accuracy'],
         label='val_accuracy',
         color='green')
plt.xlabel('No of Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.5,1)
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label ='loss', color='red')
plt.plot(history.history['val_loss'], label='val_loss', color='green')
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.legend(loc='upper right')
plt.show()

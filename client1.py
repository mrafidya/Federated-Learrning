#!/usr/bin/env python
# coding: utf-8

# # Import libraries for VGG16 based feature extraction 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.models import Model
import pickle
import sys


# # Define train and test directory 


import zipfile,os
local_zip = 're-arranged-data(1).zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()
 
base_dir = 're-arranged-data(1)/sign_data/Dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# # Define image size 

SIZE = 224


# # Create training and test data 


train_data_names = []
test_data_names = []

train_data = []
train_labels = []

for per in os.listdir(train_dir):
    #print('per coba')
    for data in glob.glob(train_dir+'/'+per+'/*.*'):
        
        train_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        train_data.append([img])
        if per[-1]=='g':
            train_labels.append(np.array(1))
        else:
            train_labels.append(np.array(0))
        #print('label coba')
train_data = np.array(train_data)/255.0
train_labels = np.array(train_labels)

#Test Data

test_data = []
test_labels = []

for per in os.listdir(test_dir):
    for data in glob.glob(test_dir+'/'+per+'/*.*'):
        test_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        test_data.append([img])
        if per[-1]=='g':
            test_labels.append(np.array(1))
        else:
            test_labels.append(np.array(0))

#print('test1')
test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)
#print('test2')

# # Prepare training and test data

# Categorical labels
#print('test3')
#print(train_labels)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#print('test4')
# Reshaping
train_data = train_data.reshape(-1, SIZE,SIZE, 3)
test_data = test_data.reshape(-1, SIZE,SIZE, 3)


# # Define model hyperparameters


input_ = (224,224,3)
EPOCHS = 10
BS = 64
output_ = 2


# # Define model, train and plot objective


base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_)

model = Sequential()
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
#The RandomRotation has a factor of 0.1 indicating the image will be rotated by 10%, around 36 degrees.
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='sigmoid'))

model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()

earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1)

early_stop=[earlyStopping]

import flwr as fl
import tensorflow as tf
# Define Flower client
class CNNClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    progess = model.fit(train_data,train_labels, batch_size=BS,epochs=EPOCHS, callbacks=early_stop,validation_split=.3)
    acc = progess.history['accuracy']
    val_acc = progess.history['val_accuracy']
    loss = progess.history['loss']
    val_loss = progess.history['val_loss']
        
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    return model.get_weights(), len(train_data), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(test_data, test_labels)
    print("Eval accuracy : ", accuracy)
    return accuracy, len(test_data), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CNNClient())

print("----------------- DONE ---------------------")
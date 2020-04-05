# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:28:45 2020

@author: Bangkit Team
"""

# IMPORT LIBRARY
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# IMPORT DATASET
dataset = pd.read_csv("https://github.com/AkhmadMuzanni/TestingBangkit/releases/download/1/cardio_train.csv")

# DATA PREPARATION
# Drop unnececsssary column
data_train = dataset.drop(["id"], 1)

# Fix strange weight value
weight = []
for w in data_train["weight"]:
  weight.append(int(w.split('.')[0]))
data_train = data_train.drop(["weight"], 1)
data_train["weight"] = weight

# Fix strange attribute values
# delete height value that above 220
data_train = data_train.drop(data_train[data_train.height > 220].index)
# delete systolic blood pressure value that above 250
data_train = data_train.drop(data_train[data_train.ap_hi > 250].index)
# delete diastolic blood pressure value that above 200
data_train = data_train.drop(data_train[data_train.ap_lo > 200].index)

# Get target label
label = data_train["cardio"]

# Drop target label and convert to numpy array
data_train = data_train.drop("cardio", 1)
label = np.array(label)
data_train = np.array(data_train)

# Min-Max Normalization Process
max_vals = np.max(data_train, 0)
data_train = data_train / max_vals

# Split Train and Test
X_train, x_test, Y_train, y_test = train_test_split(data_train, label, test_size=0.3) # 30% data test
print("Train shape:", X_train.shape)
print("Test shape:", x_test.shape)

# Build Multi Layer Perceptron (MLP) Model
# Initialize Parameter
input_dim = X_train.shape[1] # jumlah atribut
output_dim = 1 # menghasilkan output binary
lr = 0.0001
optimizer = RMSprop(learning_rate=lr)
batch_size = 512
epochs = 150

# Build Model
model = Sequential([
          # Dense layer with 100 neuron, input shape and relu activation function
          Dense(100, input_shape=(input_dim,), activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 200 neuron and relu activation function
          Dense(200, activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 100 neuron and relu activation function
          Dense(100, activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 100 neuron and sigmoid activation function
          Dense(output_dim, activation='sigmoid')
], name="cardionet")

# Visualize Model
plot_model(model,
           to_file=model.name+'.png',
           show_shapes=True,
           show_layer_names=False,
           rankdir='LR',
           dpi=70
          )

model.summary()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#Train Model
history = model.fit(X_train, Y_train, 
                    validation_data=(x_test, y_test),
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=2)

#Visualize Training Result
fig, ax = plt.subplots(1,2,figsize=(18,3))
ax[0].plot(history.history["accuracy"])
ax[0].plot(history.history["val_accuracy"])
ax[0].set_title("Train and Validation Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend(['Train', 'Validation'], loc='upper right')

ax[1].plot(history.history["loss"])
ax[1].plot(history.history["val_loss"])
ax[1].set_title("Train and Validation Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend(["Train", "Validation"], loc="upper right")
plt.show()

#Calculate Accuracy
#Predict the data test
output = model.predict(x_test)

# calculate true positive, true negative, false positive, false negative
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
for (real, pred) in zip(y_test, output):
  if pred > 0.5: # hasil prediksi benar
    if real == 1:
      true_positive += 1
    else:
      false_positive += 1
  else:
    if real == 1:
      false_negative += 1
    else:
      true_negative += 1

#Precision and Recall
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
print("Precision:", precision)
print("Recall:", recall)

#Accuracy
akurasi = (true_positive + true_negative) / output.shape[0]
print("Akurasi:", akurasi)

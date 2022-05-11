#!/usr/bin/env python
# coding: utf-8

# In[49]:


import tensorflow as tf
from tensorflow import keras

from keras import models

from keras.layers import *
import matplotlib.pyplot as plt

x = keras.datasets.mnist
(train , target_train),(test,target_test) = x.load_data()
train.shape
train = train.reshape(60000,28,28,1)
train.shape
test.shape
test = test.reshape(10000, 28, 28, 1)
test.shape
model = models.Sequential()
model.add(Conv2D(64, kernel_size = (5,5), activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPool2D())
model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(256,kernel_size=(3,3), activation = 'relu'))
model.add(MaxPool2D())

model.add(Flatten()),
model.add(Dense(128, activation='relu')),
model.add(Dense(256, activation ='relu')),
model.add(Dense(10, activation ='sigmoid'))

model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics =['accuracy'])
h = model.fit(train,target_train,epochs = 10 ,  validation_data=(test,target_test))


# In[ ]:





# In[ ]:





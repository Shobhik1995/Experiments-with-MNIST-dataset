
# coding: utf-8



# importing required packages
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt




#loading the dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()




# rolling the training data into one feature vector
fvect_size = x_train.shape[1]*x_train.shape[2]
x_train=x_train.reshape(x_train.shape[0],fvect_size)
x_test=x_test.reshape(x_test.shape[0],fvect_size)




# scaling the values of the feature vector
x_train=x_train/255
x_test=x_test/255




#one hot encoding of the labels
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]




def nn():
    model=Sequential()
    model.add(Dense(fvect_size,input_dim=fvect_size,init='uniform',activation='relu'))
    model.add(Dense(num_classes,input_dim=fvect_size,init='uniform',activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model




model=nn()
model.fit(x_train,y_train,validation_data=(x_test,y_test),nb_epoch=5,batch_size=100,verbose=2)
scores=model.evaluate(x_test,y_test,verbose=0)




print("Error : %.2f%%" % (100-scores[1]*100))









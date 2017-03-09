
# coding: utf-8

# In[67]:

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
np.random.seed(5)


# In[68]:

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

x_train=x_train/255
x_test=x_test/255
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]


# In[69]:

x_val=x_train[48001:][:][:]
y_val=y_train[48001:][:]
x_train=x_train[0:48000][:][:]
y_train=y_train[0:48000][:]


# In[70]:

def conv_net_large():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5,input_shape=(28,28,1), border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 5, 5,input_shape=(28,28,1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(128, 5,5 , border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 5,5 , border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[71]:

model=conv_net_large()
model.fit(x_train,y_train,validation_data=(x_val,y_val),nb_epoch=10,batch_size=200,verbose=2)
score=model.evaluate(x_test,y_test,verbose=0)
print("Error value : "+str(100-score[1]*100))


# In[72]:

#print("Error value : "+str(100-score[1]*100))


# In[74]:

model.save('mnist_cnn_lessdense_reg0.3.h5')


# In[ ]:




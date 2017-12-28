
# coding: utf-8

# In[49]:


import numpy as np
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10


# In[50]:


(img_train, label_train), (img_test, label_test) = cifar10.load_data()


# In[51]:


# Preprocess input data

img_train = img_train.astype('float32')
img_test = img_test.astype('float32')
img_train /= 255
img_test /= 255
print(img_train.shape)
print(label_train.shape)


# In[52]:


# Preprocess class labels
label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)
print(label_train.shape)


# In[53]:


# Define model architecture
model = Sequential()


# In[54]:


model.add(Convolution2D(32, (2, 2), strides = (1,1), activation='relu', input_shape=(32,32,3)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(64, (2, 2), strides = (1,1), activation='relu'))
model.add(Convolution2D(64, (2, 2), strides = (1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[55]:


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


# Fit model on training data
print("TRAINING")
model.fit(img_train, label_train, validation_split=0.3,
          batch_size=32, epochs=2, verbose=1)


# In[61]:


# Evaluate model on test data
print("TESTING")
score = model.evaluate(img_test, label_test, verbose=0)


# In[45]:


print(score)


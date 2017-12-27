
# coding: utf-8

# In[1]:

from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications.vgg16 import preprocess_input
from keras.layers import Embedding,GRU,TimeDistributed,Dense,RepeatVector,Merge
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.applications.vgg16 import VGG16
from keras.models import Model
#import MatplotLib
import h5py
import numpy as np
import cv2
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
import json



# In[2]:


filename = "/others/guest2/coc/coco_data.h5"

f = h5py.File(filename,'r')
print(list(f.keys())[0])   #images
image_group = list(f.keys())[0]
label_group = list(f.keys())[4]
label_len_group = list(f.keys())[2]
label_start_group = list(f.keys())[3]
label_end_group = list(f.keys())[1]

images = list(f[image_group])
label_len = list(f[label_len_group])
label_start = list(f[label_start_group])
label_end = list(f[label_end_group])
labels = list(f[label_group])
images = np.asarray(images)  ##converting list of images into numpy array



# In[3]:


input_imgcap = {}
input_imgcap['image'] = []
input_imgcap['caption_inp'] = []
input_imgcap['caption_op'] = []


# In[ ]:


print(labels[0][5])


# In[ ]:


for i,img in enumerate(images[:5000]):
    print(i)
    j = label_start[i] - 1
    while j < label_end[i]:
        img = np.array(img,dtype='float64')
        img = preprocess_input(img)
        input_imgcap['image'].append(img)
        labels[j] = list(labels[j])
        #print(len(labels[j]),labels[j])
        labels[j].insert(0,9568)  ##inserting start token for each caption
        labels[j].insert(17,9569)  ##inserting end token for each caption
        
        #labels[j] = np.array(img,dtype='float64')
        #print(labels[j])
        input_imgcap['caption_inp'].append(np.array(labels[j],dtype='float64'))
        caption = np.zeros((9569))
        #labels[j] = list(labels[j])
        for label in labels[j]:
            if label!=0:
                caption[label-1] = 1
        #print(caption)
        caption = np.array(caption,dtype = 'float64')
        input_imgcap['caption_op'].append(caption)
        #print(caption)
        j+=1





# In[ ]:


# input_imgcap['image'] = np.array(input_imgcap['image'],dtype = 'float64')
# input_imgcap['caption_inp'] = np.array(input_imgcap['caption_inp'],dtype = 'float64')
# input_imgcap['caption_op'] = np.array(input_imgcap['caption_op'],dtype = 'float64')
# np.savez_compressed('/others/guest2/coc/train_images.npz',np.array(input_imgcap['image'],dtype = 'float64'))
# np.savez_compressed('/others/guest2/coc/train_captions_inp.npz',np.array(input_imgcap['caption_inp'],dtype = 'float64'))
# np.savez_compressed('/others/guest2/coc/train_captions_op.npz',np.array(input_imgcap['caption_inp'],dtype = 'float64'))


# In[ ]:


x_train,c_train,y_train = np.array(input_imgcap['image']),np.array(input_imgcap['caption_inp']),np.array(input_imgcap['caption_op'])
# x_val,c_val,y_val = input_imgcap['image'][:25007],input_imgcap['caption_inp'][:25007],input_imgcap['caption_op'][:25007]
# x_train,c_train,y_train = input_imgcap['image'][25007:],input_imgcap['caption_inp'][25007:],input_imgcap['caption_op'][25007:]


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(c_train.shape)
print(c_train[0:5])


# In[ ]:


def vggmodel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(256,256,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    return model


# In[ ]:


max_caption_len = 18
vocab_size = 9569


# In[ ]:


#with tf.device('/gpu:0'):  
print "VGG loading"
base_model = vggmodel()
base_model.layers.pop()
base_model.trainable = False

#print(base_model.summary())
print "VGG loaded"
# let's load the weights from a save file.
# image_model.load_weights('weight_file.h5')

# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
print "Text model loading"
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(units=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))
print(language_model.summary())
print "Text model loaded"
# let's repeat the image vector to turn it into a sequence.
print "Repeat model loading"
base_model.add(RepeatVector(max_caption_len))
print "Repeat model loaded"
# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
print "Merging"
model = Sequential()
model.add(Merge([base_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print(model.summary())
model1 = multi_gpu_model(model,gpus=2)
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
print "Merged"

print "Data preprocessing done"
#history = model.fit([x_train,c_train], y_train, validation_data = ([x_val,c_val],y_val), batch_size=1, nb_epoch=5)
history = model1.fit([x_train,c_train], y_train, batch_size=64, epochs=50,verbose = 1,validation_split=0.2)

model1.save('/others/guest2/coc/model.weight.end.hdf5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/others/guest2/coc/accuracy_model.png')
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/others/guest2/coc/loss_model.png')






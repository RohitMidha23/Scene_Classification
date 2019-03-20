#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras


# In[ ]:


import tensorflow


# In[1]:


from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse 

import os
import tarfile


# In[2]:


img_size=224


# In[3]:


train=pd.read_csv('train-scene/train.csv')
test=pd.read_csv('test.csv')
image_path='train-scene/train/'


# In[4]:


import os
import matplotlib.pyplot as plot
import cv2
import numpy as np


# In[5]:


from scipy.misc import imresize
from keras.preprocessing.image import load_img
train_img=[]
for i in range(len(train)):
    temp_img=load_img(image_path+train['image_name'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)
    
train_img=np.array(train_img)
train_img=preprocess_input(train_img)


# In[6]:


test_img=[]
for i in range(len(test)):
    temp_img=image.load_img(image_path+test['image_name'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)


# In[7]:


test_img=np.array(test_img)
test_img=preprocess_input(test_img)


# In[12]:


from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

num_classes = 6

#model = Sequential()
#model.add(ResNet50(include_top=False,pooling='avg',weights='imagenet'))
#model.add(Dense(num_classes,activation='softmax'))

#model.layers[0].trainable = False

#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model = ResNet50(weights='imagenet', include_top=False)


# In[13]:


features_train=model.predict(train_img)
features_test=model.predict(test_img)


# In[15]:


train_x=features_train.reshape(17034,100352)
train_y=np.asarray(train['label'])


# In[16]:


train_y=np.asarray(train['label'])
# performing one-hot encoding for the target variable

train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)
# creating training and validation set

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.15, random_state=42)


# In[17]:


from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=100352, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='relu'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fitting the model 

model.fit(X_train, Y_train, epochs=20, batch_size=30,validation_data=(X_valid,Y_valid))


# In[18]:


validation_x=features_test.reshape(7301,100352)


# In[19]:


classes = model.predict(validation_x, batch_size=30)
class_labels = np.argmax(classes, axis=1)
class_labels_dt=pd.DataFrame(class_labels)
class_labels_dt.columns=['label']


# In[20]:


class_labels_dt.head()


# In[21]:


frames=[test,class_labels_dt]
result = pd.concat(frames,axis=1)


# In[22]:


result.head()


# In[23]:


result.to_csv('resnet50_15.csv')


# In[24]:


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_resnet50_model_15.h5'


# In[25]:


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[ ]:





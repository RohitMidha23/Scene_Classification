from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
num_classes=6
image_path='train/'

from scipy.misc import imresize
train_img=[]
for i in range(len(train)):
    temp_img=image.load_img(image_path+train['image_name'][i],target_size=(150,150))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)

test_img=[]
for i in range(len(test)):
    temp_img=image.load_img(image_path+test['image_name'][i],target_size=(150,150))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)

test_img=np.array(test_img)

y_train = keras.utils.to_categorical(train[['label']], num_classes)

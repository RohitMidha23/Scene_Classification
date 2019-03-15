from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
import pandas as pd
import matplotlib.gridspec as gridspec
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


num_classes = 6
image_size=224

directory = r'data_train'

# getting train data and label
Images = []
Labels = []
classes = ['0','1','2','3','4','5']

for class_no in classes:
    for image_file in os.listdir(os.path.join(directory,str(class_no))): #Extracting the file name of the image from Class Label folder
        image = cv2.imread(os.path.join(os.path.join(directory,str(class_no)),image_file))
        image = cv2.resize(image,(image_size,image_size))
        Images.append(image)
        Labels.append(class_no)




images = np.array(Images)
labels = np.array(Labels)

print(images.shape)
print(labels)

model = Models.Sequential()

model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu',input_shape=(image_size,image_size,3)))
model.add(Layers.Conv2D(90,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(70,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(25,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(90,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dense(25,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(num_classes,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#Utils.plot_model(model,to_file='model.png',show_shapes=True)

train = model.fit(images,labels,epochs=15,validation_split=0.20)

model.save('model1.h5')
print("Model Training Done!")
plot.plot(train.history['acc'])
plot.plot(train.history['val_acc'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(train.history['loss'])
plot.plot(train.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()
print('Plots Done!')



# getting test data
directory = r'data_test'
test_Images = []

for image_file in os.listdir(directory): #Extracting the file name of the image from Class Label folder
    image = cv2.imread(os.path.join(directory,image_file))
    image = cv2.resize(image,(image_size,image_size))
    test_Images.append(image)

test_images = np.array(test_Images)

test_labels = model.predict(test_images)

print('\npredicted image labels :\n')
print(test_labels)

test = pd.read_csv('test.csv')
test['label'] = test_labels

test.to_csv('submit.csv')

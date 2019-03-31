from fastai.vision import *
from fastai.vision.data import *
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

# train_data contains all the images and train.csv is the csv file provided

path = "train_data"
tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0.5,max_rotate = 15)
data = (ImageList.from_csv(path, csv_name = 'train.csv')
        .split_by_rand_pct()
        .label_from_df()
        .add_test_folder(test_folder = 'data_test')
        .transform(tfms, size=128)
        .databunch(num_workers=0))
data.normalize(imagenet_stats)

learn = cnn_learner(data,models.resnet152,metrics=[error_rate, accuracy])

learn.fit_one_cycle(5)

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

print(len(data.valid_ds)==len(losses)==len(idxs))
learn.save('stage1-resnet152-bs32')
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
import gc
gc.collect()

#  After getting the learning rate, the values for LR were chosen.
learn.load('stage1-resnet152-bs32')
learn.fit_one_cycle(4,max_lr = slice(5e-7,1e-5))
test = pd.read_csv('test.csv')
images = []
prediction = []
probability = []
for i in test['image_name']:
  images.append(i)
  link = str(path)+'/'+i
  img = open_image(link)
  pred_class,pred_idx,outputs = learn.predict(img)
  prediction.append(pred_class.obj)
  probability.append(outputs.abs().max().item())
answer = pd.DataFrame({'image_name':images,'label':prediction})

answer.to_csv('submission.csv')

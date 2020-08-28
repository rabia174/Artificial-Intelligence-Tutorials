# for file handling and data processing
import os
import numpy as np
import pandas as pd

# tensflow and keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing import image

# image processing
from keras_preprocessing.image import ImageDataGenerator, load_img

PATH_TO_TRAIN_DATA = 'train_cats_dogs/train'
# size of images to be generated and processed
IMAGE_SIZE = (128, 128)

# batch size and epochs
BATCH_SIZE = 50
EPOCHS = 5

# get the list of files
files = os.listdir(PATH_TO_TRAIN_DATA)
# labels
y_labels = []

# process each file one by one
for i in files:
    label = i.split('.')[0]
    y_labels.append(label)


# store the file name and label in dataframe
data = pd.DataFrame({'filename': files, 'label': y_labels})

import random
# for plotting
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
# split the data
data_train, data_test = train_test_split(data, test_size = 0.33, random_state = 42)

# grab the size of validation and train set
total_train = data_train.shape[0]
total_validate = data_test.shape[0]

train_gen = ImageDataGenerator(rotation_range = 15,
    rescale = 1./255,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1)

train_data = train_gen.flow_from_dataframe(
    # data frame containing file name and label
    data_train.iloc[:5000,:],
    # path to the directory where the data is stored
    PATH_TO_TRAIN_DATA,
    # name of column in data frame containing file name
    x_col = 'filename',
    # name of column containing label
    y_col = 'label',
    # size of image generated to be trained
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = BATCH_SIZE
)

val_gen = ImageDataGenerator(
    rescale = 1./225
)

val_data = val_gen.flow_from_dataframe(
    data_test.iloc[:500,:],
    PATH_TO_TRAIN_DATA,
    x_col = 'filename',
    y_col = 'label',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = BATCH_SIZE
)

model = Sequential()

# a convolution block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# another convolution block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# flatten the output so it can be passed to dense layers
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

# compile the model using adam for backprop
model.compile(loss = categorical_crossentropy, optimizer = 'adam',
              metrics = ['accuracy'])
#
# #
history = model.fit_generator(train_data, epochs = EPOCHS, validation_data = val_data, validation_steps = total_validate // BATCH_SIZE,
                                steps_per_epoch = total_train // BATCH_SIZE,
                                 callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)])
model.save('Dogs_cats.h5')
print("model saved: Dogs_cats.h5")

# model.load_weights('Dogs_cats.h5')
# print("model loaded: Dogs_cats.h5")

# plot the model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

score = model.evaluate_generator(val_data)

print("Total loss: {:0.5f}".format(score[0]))
print("Total accuracy: {:0.2f}".format(score[1] * 100))


img_width, img_height = IMAGE_SIZE

import shutil

folder_data = 'train_cats_dogs/'
test_dir = os.path.join(folder_data, 'train')
test_data = os.path.join(folder_data, 'test_data')
# make a directory in which we put some images for testing
os.mkdir(test_data)
names = ['cat.{}.jpg'.format(i) for i in range(1, 10)]+['dog.{}.jpg'.format(i) for i in range(1, 10)]
for name in names:
    src = os.path.join(test_dir, name)
    dst = os.path.join(test_data, name)
    shutil.copyfile(src, dst)

# Put the images in a batch holder to make it easy to predict their labels
batch_holder = np.zeros((20, img_width, img_height, 3))
for i,img in enumerate(os.listdir(test_data)):
    imge = image.load_img(os.path.join(test_data,img), target_size=(img_width,img_height))
    batch_holder[i, :] = imge

result = model.predict_classes(batch_holder)
result = np.where(result > 0, 'dog', 'cat')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, img in enumerate(batch_holder):
    fig.add_subplot(4, 5, i + 1)
    plt.title(result[i])
    plt.imshow(img / 255.)
plt.show()
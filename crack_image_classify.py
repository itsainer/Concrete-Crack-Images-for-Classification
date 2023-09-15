#%%
"""
Problem Statement:
Concrete cracks may endanger the safety and durability of a building if not being identified quickly and left untreated.

Task: Perform image classification to classify concretes with or without cracks.
"""
#%%
#IMPORT PACKAGES
import os
import datetime
import cv2
import imghdr
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# %%
#DATA PREPARATION

#File path of data images
data_dir = 'crack_data'
# %%
#Check image extension
image_exts = ['jpeg','jpg','bmp','png']

#Function: remove any images which not in image_exts
for image_class in os.listdir(data_dir):                           
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
# %%
#Pack out data
data = keras.utils.image_dataset_from_directory('crack_data')
# %%
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()        
# %%
#Show labels (0: No cracks, 1: Cracks)
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
# %%
#Scale data
#Convert value from 0-255 to 0-1
data = data.map(lambda x, y:(x/255, y))
# %%
data.as_numpy_iterator().next()
# %%
#Split data

# 0.7 means 70%, 0.2 means 20%
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)                  #test model performance, during training phase
test_size = int(len(data) * 0.1)                 #test model performance, after training phase
# %%
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
# %%
#MODEL DEVELOPMENT (Build Deep Learning model)

#Build instance
model = Sequential()
#Construct model
model.add(Conv2D(16,(3,3),1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# %%
#Compile model
model.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
#Model summary
model.summary()
#Display model structure
keras.utils.plot_model(model)
#%%
#Visualize by Tensorboard
base_log_path = r"tensorboard_logs\crack_image_classify"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(log_path)
# %%
#Train the model
early_stopping_callback = EarlyStopping('loss', patience=1)
hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tb,early_stopping_callback])
# %%
#Model Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper right')
plt.show()
#%%
#EVALUATE MODEL
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    pred = model.predict(X)
    pre.update_state(y, pred)
    re.update_state(y, pred)
    acc.update_state(y, pred)
#%%
print('Precision:',pre.result().numpy(), 'Recall:',re.result().numpy(), 'Accuracy:',acc.result().numpy())
#%%
#MODEL DEPLOYMENT

#Read image data
img = cv2.imread('img1.jpg')                      #img1, img2: original data (excluded from training)
#img = cv2.imread('img2.jpg')
plt.imshow(img)
plt.show()
#%%
#Resize image
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
#%%
#Predict using model
y_pred = model.predict(np.expand_dims(resize/255,0))
print(y_pred)
if y_pred > 0.5:
    print('Predicted image is concrete with cracks')
else:
    print('Predicted image is concrete with no cracks')
#%%
#SAVE MODEL
model.save(os.path.join('models', 'cracks_ImageClassify.h5'))
# %%

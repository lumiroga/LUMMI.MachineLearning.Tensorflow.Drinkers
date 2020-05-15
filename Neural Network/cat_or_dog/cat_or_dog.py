import cv2
import numpy as np
import os         
from random import shuffle
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Input, Dropout
from keras.optimizers import Adam
#import tflearn
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression

#tf.disable_v2_behavior()

TRAIN_DIR = './train/train'
TEST_DIR = './test1/test1'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# If dataset is not created:
#train_data = create_train_data()
#test_data = create_test_data()
# If you have already created the dataset:
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

train = train_data[:-500]
test = train_data[-500:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array([i[1] for i in test])
###############################
####### KERAS EXAMPLE #########
###############################

#tf.reset_default_graph()
#create model
#model = Sequential()
#add model layers
#model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1)))
#model.add(Conv2D(32, kernel_size=3, activation=’relu’))
#model.add(Flatten())
#model.add(Dense(10, activation=’softmax’))
#compile model using accuracy to measure model performance
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#predict first 4 images in the test set
#model.predict(X_test[:4])
###############################
########## FROM TFLEARN TO KERAS ############
###############################

model = Sequential()

model.add(Conv2D(32,strides=5,kernel_size=(IMG_SIZE,IMG_SIZE),activation='relu',input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(strides=5, dim_ordering="th"))
model.add(Conv2D(64,strides=5, kernel_size=(IMG_SIZE,IMG_SIZE),activation='relu' ))
model.add(MaxPooling2D(strides=5, dim_ordering="th"))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(rate=0.8))
model.add(Dense(2, activation="relu")) # input_shape=(24500,2

model.compile(optimizer=Adam(lr=LR),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

###############################
########## TFLEARN ############
###############################
"""
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input') 
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
"""
###########################################################


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist
import os as os
import svhn as svhn
import numpy as np
from PIL import Image
import pprint as pp
import shutil as shutil

def process_resnet_feature_data(resnet_data):
    x = []
    y = []
    for entry in resnet_data:
        img_file_name = os.path.basename(entry[0])
        y.append(img_file_name.split('.')[0])
        x.append(entry[1].reshape(2048))
    y = np_utils.to_categorical(y, 7)
    return (np.asarray(x), y)

def train(Xtrain, yTrain, XTest, yTest, input_shape=(2048,)):
    print('X_train shape:', Xtrain.shape)
    print('Y_train shape:', yTrain.shape)
    print('XTest shape:', XTest.shape)
    print('yText shape:', yTest.shape)
    
    batch_size = 128
    nb_epoch = 200
    # nb_filters = 32
    # pool_size = (2, 2)
    # kernel_size = (3, 3)

    model = Sequential()    
    model.add(Dense(128, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(yTrain.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(Xtrain, yTrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, yTest))
    score = model.evaluate(XTest, yTest, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

resnet_train_features = svhn.get_resnt50_features_data(svhn.train_bb_cropped_images)
resnet_extra_features = svhn.get_resnt50_features_data(svhn.extra_bb_cropped_images)
resnet_test_features = svhn.get_resnt50_features_data(svhn.test_bb_cropped_images)

print len(resnet_train_features)
print len(resnet_extra_features)
print len(resnet_test_features)
# print reset_train_features[0][1].shape
# print reset_train_features[:2]

# one_element = reset_train_features[0][1].reshape(2048)
# print one_element.shape
# print one_element

(x_train, y_train) = process_resnet_feature_data(resnet_train_features + resnet_extra_features)
(x_test, y_test) = process_resnet_feature_data(resnet_test_features)

print x_train.shape
print y_train.shape

print x_test.shape
print y_test.shape

print x_train[0]
print y_train[0]

train(x_train, y_train, x_test, y_test)

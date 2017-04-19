
import os
import svhn
import random
import numpy as np
from keras.preprocessing import image
import pprint as pp
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import time
from keras.datasets import mnist
from keras.layers.local import LocallyConnected2D

from keras.applications.resnet50 import preprocess_input

test_data_path = '/Users/arunkumar/ml/data/svhn/test'
train_data_path = '/Users/arunkumar/ml/data/svhn/train'
extra_data_path = '/Users/arunkumar/ml/data/svhn/extra'
valid_data_path = '/Users/arunkumar/ml/data/svhn/valid'
real_life_data_path = '/Users/arunkumar/ml/data/svhn/internet'
input_img_size = (54, 54)
crop_64_img_size = (64, 64)
crop_64_64_dir_name = 'crop_64_64'

def train(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 1000
    nb_filters = 100
    pool_size = (2, 2)
    kernel_size = (5, 5)
    nb_epoch = 10
    num_conv_layers = 4
    # dense_units_count = 3072
    dense_units_count = 256
    # drop_out = 0.1

    nb_filters = [48, 64, 128,160, 192, 192, 192, 192, 192]
    model = Sequential()
    model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(drop_out))

    for i in range(1, num_conv_layers):
        stride = (i %2) + 1 
        # print 'stride:', stride
        model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='same', activation='relu', subsample=(stride, stride)))
        # model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(drop_out))

    # print model.layers[-1].output_shape
    # model.add(LocallyConnected2D(nb_filters[-1], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(dense_units_count, activation='relu'))    
    # model.add(Dropout(drop_out))
    model.add(Dense(dense_units_count, activation='relu'))    
    # model.add(Dropout(drop_out))

    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def train_bb_cropped(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 1000
    nb_filters = 100
    pool_size = (2, 2)
    kernel_size = (5, 5)
    nb_epoch = 10
    num_conv_layers = 4
    # dense_units_count = 3072
    dense_units_count = 256
    # drop_out = 0.1

    nb_filters = [48, 64, 128,160, 192, 192, 192, 192, 192]
    model = Sequential()
    model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(drop_out))

    for i in range(1, num_conv_layers):
        stride = (i %2) + 1 
        # print 'stride:', stride
        model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='same', activation='relu', subsample=(stride, stride)))
        # model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(drop_out))

    # print model.layers[-1].output_shape
    # model.add(LocallyConnected2D(nb_filters[-1], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(dense_units_count, activation='relu'))    
    # model.add(Dropout(drop_out))
    model.add(Dense(dense_units_count, activation='relu'))    
    # model.add(Dropout(drop_out))

    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def generate_images_cropped_by_bb(path, sub_dir_name, resize_to):
    # svhn.crop_digits_and_save(path, sub_dir_name, resize_to)
    svhn.save_bounded_box_image(path, os.path.join(path, sub_dir_name), resize_to)

def get_svhn_data_files(root_path):    
    svhn_parsed_data_fileName = os.path.join(root_path,'digitStruct_parsed.pkl')
    return svhn_parsed_data_fileName

def get_data_from_bb_cropped_images(path):
    img_files = svhn.get_image_files_in_dir(path)
    random.shuffle(img_files)
    
    digits_len = []
    img_data = []
    for img_file in img_files[:10]:
        print os.path.basename(img_file)
        splitted = os.path.basename(img_file).split('.')
        the_len = int(splitted[0])
        digits_len.append(the_len)

        img = image.load_img(img_file[0], grayscale=False)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)        
        x = preprocess_input(x)
        img_data.append(x.reshape(x.shape[1],x.shape[2],x.shape[3]))

    return (np.asarray(digits_len), np.asarray(img_data))

def train1():
    train_svhn_label_file = get_svhn_data_files(train_data_path)
    test_svhn_label_file = get_svhn_data_files(test_data_path)
    (x_train, train_num_digits) = svhn.resize_and_get_svhn_data(train_svhn_label_file, input_img_size)
    (x_test, test_num_digits) = svhn.resize_and_get_svhn_data(test_svhn_label_file, input_img_size)

    print x_train.shape, train_num_digits.shape
    print x_test.shape, test_num_digits.shape


    for i in range(0,2):
        print 'Train - ', i
        train(x_train, train_num_digits, x_test, test_num_digits, (54,54,3))

def train2():
    bb_train_data_path = os.path.join(train_data_path, crop_64_64_dir_name)
    bb_test_data_path = os.path.join(test_data_path, crop_64_64_dir_name)
    bb_extra_data_path = os.path.join(extra_data_path, crop_64_64_dir_name)

    # train_svhn_label_file = get_svhn_data_files(train_data_path)
    # test_svhn_label_file = get_svhn_data_files(test_data_path)
    (x_train, train_num_digits) = get_data_from_bb_cropped_images(bb_train_data_path)
    (x_test, test_num_digits) = get_data_from_bb_cropped_images(bb_test_data_path)

    print x_train.shape, train_num_digits.shape
    print x_test.shape, test_num_digits.shape


    for i in range(0,2):
        print 'Train - ', i
        train(x_train, train_num_digits, x_test, test_num_digits, (54,54,3))

def generate_64_cropped_images():
    generate_images_cropped_by_bb(train_data_path, crop_64_64_dir_name, crop_64_img_size)
    generate_images_cropped_by_bb(test_data_path, crop_64_64_dir_name, crop_64_img_size)
    generate_images_cropped_by_bb(extra_data_path, crop_64_64_dir_name, crop_64_img_size)

# train1()
# generate_64_cropped_images()
train2()

import os
from keras import backend as K
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
# from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
# from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import time
from keras.datasets import mnist
from keras.layers.local import LocallyConnected2D
import keras
from keras.applications.resnet50 import preprocess_input

test_data_path = '/Users/arunkumar/ml/data/svhn/test'
train_data_path = '/Users/arunkumar/ml/data/svhn/train'
train_data_path = '/home/ubuntu/ml/data/svhn/train'
test_data_path = '/home/ubuntu/ml/data/svhn/test'
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

def ConvBlock(model, layers, filters, input_shape=None):
    for i in range(layers):
        if input_shape:
            model.add(Convolution2D(filters, 3, 3, activation='relu', input_shape=input_shape))
        else:    
            # model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCBlock(model):
    # model.add(Dense(4096, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))


# def create(self):
#     model = self.model = Sequential()
#     model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

#     self.ConvBlock(2, 64)
#     self.ConvBlock(2, 128)
#     self.ConvBlock(3, 256)
#     self.ConvBlock(3, 512)
#     self.ConvBlock(3, 512)

#     model.add(Flatten())
#     self.FCBlock()
#     self.FCBlock()
#     model.add(Dense(1000, activation='softmax'))

def train_bb_cropped_debug(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 128
    nb_epoch = 100    
    
    # model = Sequential()
    # ConvBlock(model, 1, 128, input_shape)
    # ConvBlock(model,1, 128,  input_shape)    
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # FCBlock(model)
    # model.add(Dense(y_train.shape[1], activation='softmax'))
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #             optimizer=keras.optimizers.Adadelta(),
    #             metrics=['accuracy'])

    model = Sequential()
    # model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

    ConvBlock(model, 2, 64, input_shape)
    ConvBlock(model, 2, 128)

    model.add(Flatten(input_shape=input_shape))
    FCBlock(model)
    FCBlock(model)

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


    print model.summary()
    
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def train_bb_cropped_mnist(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 128
    nb_epoch = 100    
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    print model.summary()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def train_bb_cropped_mnist_working(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 128
    nb_epoch = 100    
    num_conv_layers = 4
    max_pool_size = (2,2)
    conv_kernel_size = (5,5)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=conv_kernel_size,
                    activation='relu',
                    input_shape=input_shape))
    for i in range(0, num_conv_layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, kernel_size=conv_kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=max_pool_size))
        model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    print model.summary()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def train_bb_cropped_mnist_working_func_api(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 128
    nb_epoch = 100    
    num_conv_layers = 4
    max_pool_size = (2,2)
    conv_kernel_size = (5,5)
    
    inp = Input(input_shape, name="input")
    x = Conv2D(32, kernel_size=conv_kernel_size, activation='relu')(inp)

    for i in range(0, num_conv_layers):
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, kernel_size=conv_kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=max_pool_size)(x)
        x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x_digits = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model([inp], [x_digits])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])


    print model.summary()

    model.fit(x_train, [y_train], batch_size=1000, nb_epoch=100, validation_data=(x_test, [y_test]))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def train_bb_cropped_mnist_working_func_api_multi_output(x_train, y_train, train_digit1, train_digit2, train_digit3, x_test, y_test, test_digit1, test_digit2, test_digit3, input_shape, model_ouput_file):
    print 'ytest:', y_test[0]
    batch_size = 128
    nb_epoch = 1
    num_conv_layers = 4
    max_pool_size = (2,2)
    conv_kernel_size = (5,5)
    
    if os.path.exists(model_ouput_file):
        print 'Loading model from:', model_ouput_file
        model = load_model(model_ouput_file)
    else:
        inp = Input(input_shape, name="input")
        x = Conv2D(32, kernel_size=conv_kernel_size, activation='relu')(inp)
        for i in range(0, num_conv_layers):
            x = ZeroPadding2D((1, 1))(x)
            x = Conv2D(64, kernel_size=conv_kernel_size, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=max_pool_size)(x)
            x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        x_digits = Dense(y_train.shape[1], activation='softmax')(x)

        digit1 = Dense(11, activation='softmax', name='digit1')(x)
        digit2 = Dense(11, activation='softmax', name='digit2')(x)
        digit3 = Dense(11, activation='softmax', name='digit3')(x)

        model = Model([inp], [x_digits, digit1, digit2, digit3])
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        print model.summary()

    model.fit(x_train, [y_train, train_digit1, train_digit2, train_digit3], batch_size=1000, nb_epoch=nb_epoch, validation_data=(x_test, [y_test, test_digit1, test_digit2, test_digit3]))
    score = model.evaluate(x_test, [y_test, test_digit1, test_digit2, test_digit3], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print 'Saving model to:', model_ouput_file
    model.save(model_ouput_file)


def train_bb_cropped_multi_output_goodfellow(x_train, y_train, train_digit1, train_digit2, train_digit3, x_test, y_test, test_digit1, test_digit2, test_digit3, input_shape, model_ouput_file):
    print 'ytest:', y_test[0]
    batch_size = 128
    nb_epoch = 1
    num_conv_layers = 5
    max_pool_size = (2,2)
    conv_kernel_size = (5,5)
    num_conv_filter_per_layer = [64, 128,160, 192, 192, 192, 192, 192, 192]
    dense_layer_units = 3072
    zero_padding_size = (1, 1)
    
    if os.path.exists(model_ouput_file):
        print 'Loading model from:', model_ouput_file
        model = load_model(model_ouput_file)
    else:
        inp = Input(input_shape, name="input")
        x = Conv2D(48, kernel_size=conv_kernel_size, activation='relu')(inp)
        for i in range(0, num_conv_layers):
            stride = i % 2 + 1
            x = ZeroPadding2D(zero_padding_size)(x)
            x = Conv2D(num_conv_filter_per_layer[i], kernel_size=conv_kernel_size, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=max_pool_size, strides=(stride,stride))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(dense_layer_units, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(dense_layer_units, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        x_digits = Dense(y_train.shape[1], activation='softmax')(x)

        digit1 = Dense(11, activation='softmax', name='digit1')(x)
        digit2 = Dense(11, activation='softmax', name='digit2')(x)
        digit3 = Dense(11, activation='softmax', name='digit3')(x)

        model = Model([inp], [x_digits, digit1, digit2, digit3])
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        print model.summary()

    model.fit(x_train, [y_train, train_digit1, train_digit2, train_digit3], batch_size=1000, nb_epoch=nb_epoch, validation_data=(x_test, [y_test, test_digit1, test_digit2, test_digit3]))
    score = model.evaluate(x_test, [y_test, test_digit1, test_digit2, test_digit3], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print 'Saving model to:', model_ouput_file
    model.save(model_ouput_file)


def train_bb_cropped(x_train, y_train, x_test, y_test, input_shape):
    batch_size = 128
    nb_filters = 100
    pool_size = (2, 2)
    kernel_size = (5, 5)
    nb_epoch = 100
    num_conv_layers = 1
    dense_units_count = 3072
    dense_units_count = 256
    # drop_out = 0.1

    nb_filters = [48, 64, 128,160, 192, 192, 192, 192, 192]
    # model = Sequential()
    # model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape, activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(drop_out))

    # for i in range(1, num_conv_layers):
    #     stride = (i %2) + 1 
    #     # print 'stride:', stride
    #     model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='same', activation='relu', subsample=(stride, stride)))
    #     # model.add(Convolution2D(nb_filters[i], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))
    #     # model.add(MaxPooling2D(pool_size=(2, 2)))
    #     # model.add(Dropout(drop_out))

    # # print model.layers[-1].output_shape
    # # model.add(LocallyConnected2D(nb_filters[-1], kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

    # # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(dense_units_count, activation='relu'))    
    # # model.add(Dropout(drop_out))
    # model.add(Dense(dense_units_count, activation='relu'))    
    # # model.add(Dropout(drop_out))

    # model.add(Dense(y_train.shape[1]))
    # model.add(Activation('softmax'))

    # model = Sequential()
    # # model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

    # # ConvBlock(model, 2, 64, input_shape)
    # # ConvBlock(model, 2, 128)
    # # ConvBlock(model, 3, 256)
    # # ConvBlock(model, 3, 512)
    # # ConvBlock(model, 3, 512)

    # model.add(Flatten(input_shape=input_shape))
    # FCBlock(model)
    # FCBlock(model)
    # FCBlock(model)
    # model.add(Dense(y_train.shape[1], activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # print model.summary()

    # model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=1)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def generate_images_cropped_by_bb(path, sub_dir_name, resize_to):
    # svhn.crop_digits_and_save(path, sub_dir_name, resize_to)
    svhn.save_bounded_box_image(path, os.path.join(path, sub_dir_name), resize_to)

def get_svhn_data_files(root_path):    
    svhn_parsed_data_fileName = os.path.join(root_path,'digitStruct_parsed.pkl')
    return svhn_parsed_data_fileName

def get_data_from_bb_cropped_images(path, max_digits=6):
    img_files = svhn.get_image_files_in_dir(path)
    random.shuffle(img_files)
    
    digits_len = []
    img_data = []
    digit1 = []
    digit2 = []
    digit3 = []
    print img_file[:3]
    for img_file in img_files[:100]:
	#print img_file
        #print os.path.basename(img_file)
        splitted = os.path.basename(img_file).split('.')
        the_len = int(splitted[0])

        dig_cat = np_utils.to_categorical(min(max_digits, the_len), max_digits + 1)
        digits_len.append(dig_cat.reshape(dig_cat.shape[1]))

        digit1_cat = np_utils.to_categorical(int(splitted[1]), 11)
        digit1.append(digit1_cat.reshape(digit1_cat.shape[1]))

        if the_len > 1:
            digit2_cat = np_utils.to_categorical(int(splitted[2]), 11)
            digit2.append(digit2_cat.reshape(digit2_cat.shape[1]))
        else:
            digit2.append(np.zeros(11))
        
        if the_len > 2:
            digit3_cat = np_utils.to_categorical(int(splitted[3]), 11)
            digit3.append(digit3_cat.reshape(digit3_cat.shape[1]))
        else:
            digit3.append(np.zeros(11))
        
        img = image.load_img(img_file, grayscale=False)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)        
        # x = preprocess_input(x)
        img_data.append(x.reshape(x.shape[1],x.shape[2],x.shape[3]))
    
    img_data = np.asarray(img_data)
    img_data = img_data.astype('float32')
    img_data /= 255
    return (img_data, np.asarray(digits_len), np.asarray(digit1), np.asarray(digit2), np.asarray(digit3))

def train1():
    train_svhn_label_file = get_svhn_data_files(train_data_path)
    test_svhn_label_file = get_svhn_data_files(test_data_path)
    (x_train, train_num_digits) = svhn.resize_and_get_svhn_data(train_svhn_label_file, input_img_size)
    (x_test, test_num_digits) = svhn.resize_and_get_svhn_data(test_svhn_label_file, input_img_size)

    print x_train.shape, train_num_digits.shape
    print x_test.shape, test_num_digits.shape

    print train_num_digits[:10]


    for i in range(0,2):
        print 'Train - ', i
        train(x_train, train_num_digits, x_test, test_num_digits, (54,54,3))

def train2():
    bb_train_data_path = os.path.join(train_data_path, crop_64_64_dir_name)
    bb_test_data_path = os.path.join(test_data_path, crop_64_64_dir_name)
    bb_extra_data_path = os.path.join(extra_data_path, crop_64_64_dir_name)

    (x_train, train_num_digits, train_digit1, train_digit2, train_digit3) = get_data_from_bb_cropped_images(bb_train_data_path)
    (x_test, test_num_digits, test_digit1, test_digit2, test_digit3) = get_data_from_bb_cropped_images(bb_test_data_path)

    print x_train.shape, train_num_digits.shape
    print x_test.shape, test_num_digits.shape
    # print x_train[0]
    # print train_num_digits[0]

    # print (x_train, train_num_digits)

    for i in range(0,1):
        print 'Train - ', i
        train_bb_cropped_multi_output_goodfellow(x_train, train_num_digits, train_digit1, train_digit2, train_digit3, x_test, test_num_digits, test_digit1, test_digit2, test_digit3, (64,64,3), 'train_bb_cropped_multi_output_goodfellow.h5')

def generate_64_cropped_images():
    generate_images_cropped_by_bb(train_data_path, crop_64_64_dir_name, crop_64_img_size)
    generate_images_cropped_by_bb(test_data_path, crop_64_64_dir_name, crop_64_img_size)
    generate_images_cropped_by_bb(extra_data_path, crop_64_64_dir_name, crop_64_img_size)


print K.image_dim_ordering()

# train1()
# generate_64_cropped_images()
for i in range(1,100):
    print 'Run:', i
    train2()

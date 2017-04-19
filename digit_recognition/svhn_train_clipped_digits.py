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

model_name_digits_clipped = 'svhn_name_digits_clipped_model.h5'
model_name_digits_clipped = 'svhn_name_digits_clipped_model_no_padding.h5'
model_name_digits_clipped = 'svhn_name_digits_clipped_model_1_padding.h5'
model_name_digits_clipped = 'svhn_name_digits_clipped_model_1_padding_morelayers.h5'
model_name_digits_clipped = 'svhn_name_digits_clipped_model_with_extra_data.h5'
model_name_digits_clipped_conv_3d = 'svhn_name_digits_clipped_conv_3d_model.h5'
wrong_predictions_directory = '/Users/arunkumar/temp/wrong_predictions'


def train(Xtrain, yTrain, XTest, yTest, inpShape, model_path, model_number):
    print('X_train shape:', Xtrain.shape)
    print('Y_train shape:', y_train.shape)
    print('XTest shape:', XTest.shape)
    print('yText shape:', yTest.shape)
    print ('inp_shape', inpShape)
    
    batch_size = 128
    nb_epoch = 5
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    if (os.path.exists(model_path)):
        print 'Model exists. Loading', model_path
        model = load_model(model_name_digits_clipped)
        #print model.summary()
    elif model_number ==1:
        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=inpShape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(yTrain.shape[1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    elif model_number ==2:
        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=inpShape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(yTrain.shape[1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(Xtrain, yTrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, yTest))
    score = model.evaluate(XTest, yTest, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print 'Saving model:', model_name_digits_clipped
    model.save(model_name_digits_clipped)
    del model


def getTrainData():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print (X_train.shape, X_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 3)
    input_shape = (3, X_train.shape[1], X_train.shape[2])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test ), input_shape

def make_predictions(model_path, test_x, test_y, test_files):
    print 'Making predictions:', test_x.shape, test_y.shape
    model = load_model(model_path)
    predictions = model.predict(test_x, 32, verbose=0)
    print len(predictions)
    print len(test_y)
    actual = [np.argmax(p) for p in test_y]
    predicted = [np.argmax(p) for p in predictions]

    for i in range(0, len(actual)):
        if actual[i] != predicted[i]:
            print actual[i], predicted[i], test_files[i]
            print os.path.basename(test_files[i])
            file_name_no_path = os.path.basename(test_files[i])
            splt = file_name_no_path.split('.')
            new_file_name = '{}.{}.{}'.format('.'.join(splt[:-1]), predicted[i], splt[-1])
            dest_file = os.path.join(wrong_predictions_directory, new_file_name)
            print 'Copying', test_files[i], 'to', dest_file
            # copy(test_files[i], dest_file)
            shutil.copyfile(test_files[i], dest_file)
            


(X_train, y_train), (X_test, y_test), inp_shape = svhn.get_svhn_data()
print 'Starting training'
print X_train.shape
print 'inp_shape:', inp_shape

for i in range(0,20):
    #train(X_train, y_train, X_test, y_test, inp_shape, model_name_digits_clipped)
    #train(X_train, y_train, X_train, y_train, inp_shape, model_name_digits_clipped)
    train(X_train, y_train, X_test, y_test, inp_shape, model_name_digits_clipped, 2)
    # train_conv_3d(X_train, y_train, X_test, y_test, inp_shape, model_name_digits_clipped_conv_3d)

# m = load_model(model_name_digits_clipped)
# print 'loading from ', model_name_digits_clipped
# print m.summary()
(test_data, test_y ), inp_shape, test_files = svhn.get_small_test_set(100)
make_predictions(model_name_digits_clipped, test_data, test_y, test_files)

# pp.pprint(test_files)

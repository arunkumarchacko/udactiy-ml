from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 2

char_sequence_length = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

def concatenate_digits(startIndex, length, data, axis):
    return np.concatenate(data[startIndex:startIndex + length], axis=axis)

def get_training_data(number_of_digits, rows, cols):
    (X, y), (X_t, y_t) = mnist.load_data()

    y = np_utils.to_categorical(y, nb_classes)
    y_t = np_utils.to_categorical(y_t, nb_classes)

    print ('X shape:', X.shape, 'X_t shape:', X_t.shape)
    if K.image_dim_ordering() == 'th':
        print ('test:', concatenate_digits(5, 3, X))
        X = X.reshape(X.shape[0], 1, rows, cols)
        X_t = X_t.reshape(X_t.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        X = X.reshape(X.shape[0], rows, cols, 1)
        X_t = X_t.reshape(X_t.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

        X_combined = np.array([concatenate_digits(i, char_sequence_length, X, 1) for i in range(X.shape[0] - number_of_digits - 1)])
        X_combined_test = np.array([concatenate_digits(i, char_sequence_length, X, 1) for i in range(X_t.shape[0] - number_of_digits - 1)])

        y_combined = np.array([concatenate_digits(i, char_sequence_length, y, 0) for i in range(y.shape[0] - number_of_digits - 1)])
        y_combined_test = np.array([concatenate_digits(i, char_sequence_length, y_t, 0) for i in range(y_t.shape[0] - number_of_digits - 1)])

        print (type(X), type(X_combined))

    X = X.astype('float32')
    X_t = X_t.astype('float32')
    X /= 255
    X_t /= 255

    print ('input_shape', input_shape)
    print('X_train shape:', X.shape)
    print(X.shape, 'train samples')
    print(X_combined.shape, 'combined train samples')
    print(X_combined_test.shape, 'combined test train samples')
    print(y_combined.shape, 'y combined train samples')
    print(y_combined_test.shape, 'y combined test train samples')
    print(X_t.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    
    
    print ('YTrain[0]', y[0])
    print ('YTest[0]',y_t[0])

    return (X, y), (X_t, y_t), input_shape

(X_train, Y_train), (X_test, Y_test), input_shape = get_training_data(3, img_rows, img_cols)

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# print ('YTrain[0]', Y_train[0])
# print ('YTest[0]',Y_test[0])



print(concatenate_digits(5,4, X_train).shape)

def train(outputDepth, rowCount, colCount):
    model = Sequential()

    
    model.add(Convolution2D(outputDepth, rowCount, colCount, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

train(nb_filters, kernel_size[0], kernel_size[1])
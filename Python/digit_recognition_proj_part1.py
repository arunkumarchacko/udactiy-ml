# Get a model for 28*28

# Get training data by concatenating images
# Get testing data by concantenating images

# Get 28*28 images by sliding a window with stride 4
# Get classification for each of them. For three digits there will be 150 one-hot encoded vector from this
# Have 3 (number of digits) classifiers trained on the 150 encoded features
# To output concatinate the outputs of each of the classifiers.abs

from __future__ import print_function
import os
import numpy as np
import pprint as pp
import random

from keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

single_digit_model_filename = 'single_digit_recognizer_model.h5'
def concatenate_digits(startIndex, length, data, axis):
    return np.concatenate(data[startIndex:startIndex + length], axis=axis)

def concatenate_sliding_window(data, length, axis):
    return np.array([concatenate_digits(i, length, data, axis) for i in range(data.shape[0] - length)])

def train_and_get_one_digit_recognizer(Xtrain, yTrain, XTest, yTest, inp_shape):
    batch_size = 128
    nb_epoch = 10
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
    XTest = XTest.reshape(XTest.shape[0], XTest.shape[1], XTest.shape[2], 1)
    
    Xtrain = Xtrain.astype('float32')
    XTest = XTest.astype('float32')
    Xtrain /= 255
    XTest /= 255

    if os.path.exists(single_digit_model_filename):
        print('Found model, loading from disk')
        model = load_model(single_digit_model_filename)
        score = model.evaluate(XTest, yTest, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return model
    
    print('Model not yet present. Creating and training model')
    print ('Xtrain after reshape:', Xtrain.shape, 'Xtest after reshape:', XTest.shape)
    print (Xtrain.shape, yTrain.shape, XTest.shape, yTest.shape, inp_shape)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=inp_shape))
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

    model.fit(Xtrain, yTrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, yTest))
    score = model.evaluate(XTest, yTest, verbose=0)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print('Saving model:', single_digit_model_filename)
    model.save(single_digit_model_filename)

    return model


def get_concatenated_digits(X_train, y_train, X_test, y_test, num_digits):
    x_tr = concatenate_sliding_window(X_train, num_digits, 1)
    x_tst = concatenate_sliding_window(X_test, num_digits, 1)

    y_tr = concatenate_sliding_window(y_train, num_digits, 0)
    y_tst = concatenate_sliding_window(y_test, num_digits, 0)

    return (x_tr, y_tr), (x_tst, y_tst)

def getClippedImages(img, resultImageWidth, stride):
    #print (img.shape, resultImageWidth, stride)
    #return [img[:,theIndex:theIndex + resultImageWidth, :]  for theIndex in range(0, img.shape[1] - resultImageWidth + 1, stride)]
    return [img[:,theIndex:theIndex + resultImageWidth]  for theIndex in range(0, img.shape[1] - resultImageWidth + 1, stride)]

def run_single_digit_recognizer(images, image_cols, sliding_window, recognizer):
    combined_result = []
    for index in range(images.shape[0]):
        result = getClippedImages(images[index], image_cols, sliding_window)
        #print ('Number of clipped images:', len(result), result[0].shape)
        combined = np.dstack(result)        
        combined = np.rollaxis(combined, -1)
        shp = combined.shape
        combined = combined.reshape(shp[0], shp[1], shp[2], 1)
        #print ('Combined shape:', combined.shape)
        digit = recognizer.predict(combined)
        
        #aa = [np.argmax(d) for d in digit]
        #print (aa)
        # print (digit.shape)
        # print (digit.flatten().shape)
        combined_result.append(digit.flatten())    
    features = np.array(combined_result)
    #features = np.rollaxis(features, -1)
    return features

def train_model_for_digit(X_train, Y_train, X_test, Y_test, theshape):
    print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, theshape)
    # print_result(X_train[0], 10)
    # print_result(Y_train[0], 10)

    # print_result(X_train[1], 10)
    # print_result(Y_train[1], 10)

    batch_size = 128
    nb_epoch = 5
    
    # Y_train = Y_train[2:,]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)

    print ('train_model_for_digit:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, theshape)
    model = Sequential()    
    model.add(Dense(128, input_shape=theshape, name='dense1'))
    model.add(Activation('relu', name='activation1'))
    model.add(Dropout(0.5))
    # model.add(Dense(Y_train.shape[1], name='dense2'))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    # model.add(Activation('relu'))
    # model.add(Dense(64))
    # model.add(Dropout(0.5))
    # model.add(Dense(32))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax', name='activation2'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])    

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model
    

def get_label_for_digit(fullLabel, digit_index, num_digits):
    print('Getting label for digit at index:', digit_index, fullLabel.shape)
    result = []
    for i in range(0, fullLabel.shape[0]):
        row = fullLabel[i]
        sample = np.zeros(num_digits)
        start = digit_index*num_digits
        for i in range(start, start + num_digits):
            sample[i-start] = row[i]
        result.append(sample)
    return np.array(result)

def print_result(result, cols):
    print('-------------')
    for i in range(0, len(result), cols):
        print (' '.join(str(v) for v in result[i:i+cols]))        

def predict_digits(models, to_predict, sliding_window, image_cols, one_digit_recognizer):
    features = run_single_digit_recognizer(to_predict, image_cols, sliding_window, one_digit_recognizer)    
    features = features.reshape(features.shape[0], features.shape[1], 1)
    # print ('to_predict:', to_predict.shape, features.shape)
    # print_result(features, 10)
    for m in models:
        for p in m.predict(features):
            print (np.argmax(p))

def get_empty_image(size):
    result = np.zeros(size * size)
    for i in range(result.shape[0]):
        result[i] = random.randint(0,20)
    return result.reshape(size, size)

num_digits = 5
sliding_window = 2
image_cols = 28
inp_shape = (28,28, 1)
#classifier_inp_shape = (num_digits * 10, 1)
n_classes = 11
train_size = 1500

print ('Loading data')
(X_train, y_train), (X_test, y_test) = mnist.load_data()
pp.pprint(X_train[0])
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

for i in range(X_train.shape[0] / 10):
    empty_img = get_empty_image(image_cols)
    pp.pprint(empty_img)
    break

print ('Training one-digit recognizer')
one_digit_recognizer = train_and_get_one_digit_recognizer(X_train[:train_size], y_train[:train_size], X_test[:train_size], y_test[:train_size], inp_shape)

print (one_digit_recognizer.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)[0:4]))
print ('Getting concatenated digits')
(X_train_multiple, y_train_multiple), (X_test_multiple, y_test_multiple) = get_concatenated_digits(X_train[:train_size], y_train[:train_size], X_test[:train_size], y_test[:train_size], num_digits)
print (X_train_multiple.shape, y_train_multiple.shape, X_test_multiple.shape, y_test_multiple.shape)

print ('Running single digit recognizer on training data')
X_train_features = run_single_digit_recognizer(X_train_multiple, image_cols, sliding_window, one_digit_recognizer)
print(X_train_features.shape)
print(y_train_multiple.shape)
# pp.pprint(X_train_features[0])
# pp.pprint(y_train_multiple[0])
# print_result(X_train_features[0], 10)
# print_result(y_train_multiple[0], 10)

# print_result(X_train_features[1], 10)
# print_result(y_train_multiple[1], 10)

classifier_inp_shape = (X_train_features.shape[1], 1)
print ('Running single digit recognizer on testing data')
X_test_features = run_single_digit_recognizer(X_test_multiple, image_cols, sliding_window, one_digit_recognizer)
print(X_test_features.shape)

print ('Training classifiers for each digits')
#models = [train_model_for_digit(X_train_features, get_label_for_digit(y_train_multiple, i, n_classes), X_test_features, get_label_for_digit(y_test_multiple, i, n_classes), classifier_inp_shape)  for i in range(0, num_digits)]
models = [train_model_for_digit(X_train_features, get_label_for_digit(y_train_multiple, i, n_classes), X_test_features, get_label_for_digit(y_test_multiple, i, n_classes), classifier_inp_shape)  for i in range(0, num_digits)]

print ('Models:', len(models))

# predict_digits(models, np.array([X_train_multiple[0]]), sliding_window, image_cols, one_digit_recognizer)
predict_digits(models, np.array([X_test_multiple[0]]), sliding_window, image_cols, one_digit_recognizer)


# (497, 50, 1)     (495, 10) (497, 50, 1)     (497, 10) (50, 1)
# (500, 28, 28, 1) (500, 10) (500, 28, 28, 1) (500, 10) (28, 28, 1)
# 721041495906
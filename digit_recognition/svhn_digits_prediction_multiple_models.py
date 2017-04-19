
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist
import os as os
import svhn as svhn
import numpy as np
from PIL import Image
import pprint as pp
import shutil as shutil
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model, Sequential, load_model
from random import shuffle

max_digits = 7
one_hot_bits = 11
predict_numdigits_and_digits_4_digits_model = "predict_numdigits_and_digits_4_digits_model.h5"

def get_onehot_withdigit(digit, max_digit): 
    result = np.zeros(max_digit)
    result[digit] = 1
    return result

def process_resnet_feature_data(resnet_data):
    x = []
    y = []
    
    for i in range(0, max_digits + 1):
        y.append([])

    for entry in resnet_data:
        x.append(entry[1].reshape(2048))
        # y_out = []
        img_file_name = os.path.basename(entry[0])
        # print img_file_name
        split_digits = img_file_name.split('.')
        number_of_digits = int(split_digits[0])
        # y_out.append(get_onehot_withdigit(number_of_digits, max_digits))

        y[0].append(get_onehot_withdigit(number_of_digits, max_digits))

        for i in range(0, max_digits):
            if i >= number_of_digits:
                y[i+1].append(np.zeros(one_hot_bits))
            else:
                y[i+1].append(get_onehot_withdigit(int(split_digits[i+1]), one_hot_bits))
        
        # pp.pprint(y_out)

        # y.append(np.asarray(y_out))
    # y = np_utils.to_categorical(y, 7)
    # print y

    for i in range(0, len(y)):
        y[i] = np.asarray(y[i])

    return (np.asarray(x), y)

def train(Xtrain, yTrain, XTest, yTest, input_shape, model_path): 
    print('X_train shape:', Xtrain.shape)
    # print('Y_train shape:', yTrain.shape)
    print('XTest shape:', XTest.shape)
    # print('yText shape:', yTest.shape)
    
    batch_size = 128
    nb_epoch = 10
    # nb_filters = 32
    # pool_size = (2, 2)
    # kernel_size = (3, 3)

    # model = Sequential()    
    # model.add(Dense(128, input_shape=input_shape))
    # model.add(Activation('relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # # model.add(Dropout(0.75))
    # model.add(Dense(yTrain[0].shape[1]))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # model.fit(Xtrain, yTrain[0], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, yTest[0]))
    # score = model.evaluate(XTest, yTest, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])   


    # inputs = Input(shape=input_shape)
    # x = Dense(128, activation='relu')(inputs)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(max_digits, activation='softmax')(x)
    
    if (os.path.exists(model_path)):
        print 'Model exists. Loading', model_path
        model = load_model(model_path)
    else:
        a = Input(shape=input_shape)
        b = Dense(256, activation='relu', name="d1")(a)
        b = Dense(512, activation='relu', name="d11")(a)
        num_digits = Dense(128, activation='relu', name="d2")(b)
        digits = Dense(128, activation='relu', name="d3")(b)
        num_digits = Dense(max_digits, activation='softmax', name="d4")(num_digits)
        digits = Dense(512, activation='relu', name="d5")(digits)
        digits = Dense(512, activation='relu')(digits)
        digit1 = Dense(one_hot_bits, activation='softmax', name="digit1")(digits)
        digit2 = Dense(one_hot_bits, activation='softmax', name="digit2")(digits)
        digit3 = Dense(one_hot_bits, activation='softmax', name="digit3")(digits)
        digit4 = Dense(one_hot_bits, activation='softmax', name="digit4")(digits)
        model = Model(input=a, output=[num_digits, digit1, digit2, digit3, digit4])
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(Xtrain, [yTrain[0], yTrain[1], yTrain[2], yTrain[3], yTrain[4]], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, [yTest[0],yTest[1], yTest[2],yTest[3], yTest[4]]))

    print('Saving to', model_path)
    model.save(model_path)

    # print('X_train shape:', Xtrain.shape)
    # # print('Y_train shape:', yTrain.shape)
    # print('XTest shape:', XTest.shape)
    # # print('yText shape:', yTest.shape)

    # pp.pprint(yTrain[0][:10])
    # pp.pprint(yTrain[0][25000:25050])
    # pp.pprint(yTest[0][:10])
    
    # batch_size = 128
    # nb_epoch = 20
    # # nb_filters = 32
    # # pool_size = (2, 2)
    # # kernel_size = (3, 3)

    # inputs = Input(shape=(2048,))
    # x = Dense(2048, activation='relu')(inputs)
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # num_digits = Dense(max_digits, activation='softmax', name="num1")(x)
    # # x = Dense(2048, activation='relu')(x)
    # # x = Dense(1024, activation='relu')(x)
    # # x = Dense(256, activation='relu')(x)
    # # digit1 = Dense(one_hot_bits, activation='softmax', name="d1")(x)
    # # digit2 = Dense(one_hot_bits, activation='softmax', name="d2")(x)
    # # digit3 = Dense(one_hot_bits, activation='softmax', name="d3")(x)
    # # digit4 = Dense(one_hot_bits, activation='softmax', name="d4")(x)
    # # digit5 = Dense(one_hot_bits, activation='softmax', name="d5")(x)
    # # digit6 = Dense(one_hot_bits, activation='softmax', name="d6")(x)
    # # digit7 = Dense(one_hot_bits, activation='softmax', name="d7")(x)

    # # model = Model(input=inputs, output=[num_digits, digit1, digit2, digit3, digit4, digit5, digit6, digit7])
    # model = Model(input=inputs, output=num_digits)
    
    # model.compile(optimizer='rmsprop',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])
    # # model.fit(data, labels)  # starts training

    # print model.summary()
    # model.fit(Xtrain, yTrain[0], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(XTest, yTest[0]))
    # score = model.evaluate(XTest, yTest[0], verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

resnet_train_features = svhn.get_resnt50_features_data(svhn.train_bb_cropped_images)
resnet_extra_features = svhn.get_resnt50_features_data(svhn.extra_bb_cropped_images)
resnet_test_features = svhn.get_resnt50_features_data(svhn.test_bb_cropped_images)

shuffle(resnet_test_features)
shuffle(resnet_extra_features)

print len(resnet_train_features)
print len(resnet_extra_features)
print len(resnet_test_features)
# print reset_train_features[0][1].shape
# print reset_train_features[:2]

# one_element = reset_train_features[0][1].reshape(2048)
# print one_element.shape
# print one_element

(x_train, y_train) = process_resnet_feature_data(resnet_train_features + resnet_extra_features)
print "yTrain"
# pp.pprint(y_train)
pp.pprint(y_train[0])
pp.pprint(y_train[1])
pp.pprint(y_train[2])

#(x_test, y_test) = process_resnet_feature_data(resnet_test_features[:10])
(x_test, y_test) = process_resnet_feature_data(resnet_test_features)

print x_train.shape
# print y_train.shape

print x_test.shape
# print y_test.shape

# print x_train[0]
# print y_train[0]

# print "yTest"
# pp.pprint(y_test)

for i in range(0,1000):
    train(x_train, y_train, x_test, y_test, (2048,), predict_numdigits_and_digits_4_digits_model)

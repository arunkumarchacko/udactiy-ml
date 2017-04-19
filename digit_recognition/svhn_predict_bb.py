import os
import svhn
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
# from keras.preprocessing import np_utils
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import time


models_dir_name = "models"
# renet50_convolution_features = 'renet50_convolution_features.h5' 
renet50_convolution_features = 'renet50_convolution_features_224_224.h5' 
test_data_path = '/Users/arunkumar/ml/data/svhn/test'
test_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/test/digitStruct_parsed.pkl'
test_model_dir = os.path.join(test_data_path, models_dir_name)
test_resnet_features_full_path = os.path.join(test_model_dir, renet50_convolution_features)

train_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/train/digitStruct_parsed.pkl'
train_data_path = '/Users/arunkumar/ml/data/svhn/train'
train_model_dir = os.path.join(train_data_path, models_dir_name)
train_resnet_features_full_path = os.path.join(train_model_dir, renet50_convolution_features)

extra_train_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/extra/digitStruct_parsed.pkl'
extra_data_path = '/Users/arunkumar/ml/data/svhn/extra'
extra_model_dir = os.path.join(extra_data_path, models_dir_name)
extra_resnet_features_full_path = os.path.join(extra_model_dir, renet50_convolution_features)

# model_output = "bb_and_numdigits_predictor.h5"
model_output = "bb_and_numdigits_predictor_224_224.h5"

def generate_resnet50_for_path(path, file_name):
    dir_name = os.path.join(path, models_dir_name)
    if not os.path.exists(dir_name):
        print('Creating:', dir_name)
        os.mkdir(dir_name)

    out_filename = os.path.join(dir_name, file_name)
    # svhn.generate_features_using_resnet_and_save(path, 1000, out_filename, (200,480))
    return svhn.generate_features_using_resnet_and_save(path, 1000, out_filename, (224,224))
    
def load_resnet_features(file_name):
    result = {}    
    for item in svhn.load_pickle_file(file_name):
        print 'item:', item
        result[item[0]] = item
    return result
    
def convert_bb(bb_data, size):
    
    # bb = [bb[p] for p in bb_params]
    # pp.pprint(bb_data)
    # print svhn.get_image_size(bb_data[0])
    bb = []
    f = bb_data[1]
    bb.append(f[0])
    bb.append(f[1])
    bb.append(f[2])
    bb.append(f[3])

    # conv_x = (480. / size[0])
    # conv_y = (200. / size[1])
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    # print 'Converting:', bb_data, size, conv_x, conv_y
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_y, 0)
    bb[3] = max(bb[3]*conv_x, 0)

    # print 'Converted:', bb, size
    return bb

# trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)
# val_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)
                   
def get_data_for_train(features_file_name, bb_data_file_name, extra_features_file_name=None, extra_bb_data_file_name=None):
    print 'Features File:', features_file_name
    print 'bb_data File:', bb_data_file_name
    print 'Extra Features File:', extra_features_file_name
    print 'Extra bb_data File:', extra_bb_data_file_name

    bb_data = svhn.get_all_digits_bb_data(bb_data_file_name)
    test_features = load_resnet_features(features_file_name)

    if extra_bb_data_file_name:
        bb_data += svhn.get_all_digits_bb_data(extra_bb_data_file_name)
        for k,v in load_resnet_features(features_file_name).iteritems():
            test_features[k] = v

    print 'bb_data Count:', len(bb_data), 'FeaturesCount:', len(test_features.keys())

    # pp.pprint(bb_data[100:103])
    print 'Features'
    pp.pprint(test_features.keys()[:3])
    print len(test_features)
    

# (top, left, bottom, right, label)
    x = []
    top = []
    left = []
    bottom = []
    right = []
    label = []

    bb_data = bb_data[0:15]
    trn_bbox = np.stack([convert_bb(bbd, svhn.get_image_size(bbd[0])) for bbd in bb_data]).astype(np.float32)
    lbs = np.stack(np_utils.to_categorical(bbd[1][4].split('.')[0], 8).reshape(8) for bbd in bb_data)
    file_names = [b[0] for b in bb_data]

    # svhn.plot_image_from_path(bb_data[0][0], trn_bbox[0], target_size=(224,224))
    for bbd in bb_data:
        # print 'bbd:', bbd
        # svhn.plot_image_from_path(bbd[0], bbd[1])
        # print test_features[bbd[0]][1].shape
        x.append(test_features[bbd[0]][1].reshape(1,2048))
    #     f = bbd[1]
    #     top.append(f[0])
    #     left.append(f[1])
    #     bottom.append(f[2])
    #     right.append(f[3])
    #     label.append(f[4].split('.')[0])
        
    return (np.asarray(x), trn_bbox, lbs, file_names)

def predict_and_display(conv_feat, model_path, file_names):
    print 'Predicting'
    if os.path.exists(model_path):
        print 'Loading model from:', model_path
        model = load_model(model_path)
        pred = model.predict(conv_feat[0:10])
        # print pred[0]
        # print np.argmax(pred[1])

        for i, bb in enumerate(pred[0]):
            print i, bb
            svhn.plot_image_from_path(file_names[i], bb, target_size=(224,224))


def generate_intermediate_features():
    train_done = False
    test_done = False
    extra_done = False
    for i in range(1,300):
        if (not train_done):
            train_done = generate_resnet50_for_path(train_data_path, renet50_convolution_features)
        
        if (not test_done):
            test_done = generate_resnet50_for_path(test_data_path, renet50_convolution_features) 
        
        if train_done and not extra_done:
            extra_done = generate_resnet50_for_path(extra_data_path, renet50_convolution_features)

        if train_done and test_done and extra_done:
            print "Train, Test and Extra processing done."
            break
    

def train(inp_shape, trn_bbox, trn_labels, conv_feat, conv_val_feat, val_bbox, val_labels, out_path):
    print conv_feat.shape
    print inp_shape
    if os.path.exists(out_path):
        print 'Loading model from:', out_path
        model = load_model(out_path)
    else:
        inp = Input(inp_shape, name="input")
        # x = MaxPooling2D()(inp)
        # x = BatchNormalization(axis=1)(x)
        # x = Dropout(p/4)(x)
        # x = Flatten()(x)
        x = Dense(1024, activation='relu')(inp)
        x = Dropout(0.5)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(p)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        # x = BatchNormalization()(x)
        # x = Dropout(p/2)(x)
        x_bb = Dense(4, name='bb')(x)
        x_class = Dense(8, activation='softmax', name='numDigits')(x)

        model = Model([inp], [x_bb, x_class])
        model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
                loss_weights=[.001, 1.])
    model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=1000, nb_epoch=2, validation_data=(conv_val_feat, [val_bbox, val_labels]))
    model.save(out_path)

    # pred = model.predict(conv_val_feat[:10])
    # print 'PredLabels:', pred[1]
    # print 'PredBB:', pred[0]

def performTraining():
    startTime = time.time()
    print 'Starting training'
    (x_train, y_train_bb, y_train_labels, train_files) = get_data_for_train(train_resnet_features_full_path, train_svhn_parsed_data_fileName, extra_resnet_features_full_path, extra_train_svhn_parsed_data_fileName)
    (x_test, y_test_bb, y_test_labels, test_files) = get_data_for_train(test_data_path, renet50_convolution_features, test_svhn_parsed_data_fileName)
    # (x_extra_train, y_extra_train_bb, y_extra_train_labels, extra_files) = get_data_for_train(extra_data_path, renet50_convolution_features, extra_train_svhn_parsed_data_fileName)


    pp.pprint(x_test.shape)
    print x_train.shape, y_train_bb.shape, y_train_labels.shape

    print x_train[0]
    print train_files[:10]
    print test_files[:10]

    # for i in range(0,2):
    #     train((1,2048), y_train_bb, y_train_labels, x_train, x_test, y_test_bb, y_test_labels, model_output)
    
    # pp.pprint(y_test)
    # pp.pprint(y_test[:5])

    print "Training Elapsed:", time.time() - startTime

generate_intermediate_features()
# performTraining()


# svhn.get_image_size_in_dir(test_data_path)

# img = image.load_img('/Users/arunkumar/ml/data/svhn/test/6234.png', target_size=(372, 1083))
# img.show()

# img1 = image.load_img('/Users/arunkumar/ml/data/svhn/test/6234.png', target_size=None)
# img1.show()

# predict_and_display(x_train, model_output, train_files)
# predict_and_display(x_test, model_output, test_files)
# predict_and_display(x_extra_train, model_output, extra_files)
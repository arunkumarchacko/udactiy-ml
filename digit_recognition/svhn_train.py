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
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import time

test_data_path = '/Users/arunkumar/ml/data/svhn/test'
train_data_path = '/Users/arunkumar/ml/data/svhn/train'
extra_data_path = '/Users/arunkumar/ml/data/svhn/extra'
valid_data_path = '/Users/arunkumar/ml/data/svhn/valid'
bb_predictor_model_output = "bb_and_numdigits_predictor_model_1.h5"
end_end_predictor_model_output_2_digits = "end_end_predictor_model_2_digit.h5"
end_end_predictor_model_output_3_digits = "end_end_predictor_model_3_digts.h5"
end_end_predictor_model_output_2_digits_no_drop_out = "end_end_predictor_model_2_digit_no_drop_out.h5"
end_end_predictor_model_output_3_digits_no_drop_out = "end_end_predictor_model_3_digts_no_drop_out.h5"

real_life_data_path = '/Users/arunkumar/ml/data/svhn/internet'
conv_feature_file_name =  "conv_feature_data_24_24.pkl"
# conv_feature_file_name =  'renet50_convolution_features_224_224.h5' 

def get_data_files(root_path):
    renet50_convolution_features = conv_feature_file_name
    svhn_parsed_data_fileName = os.path.join(root_path,'digitStruct_parsed.pkl')
    model_dir = os.path.join(root_path, 'models')
    resnet_features_full_path = os.path.join(model_dir, renet50_convolution_features)

    return (svhn_parsed_data_fileName, resnet_features_full_path)

def generate_conv_features(dir_path, feat_filename, count_to_process=1000):
        feat_dir = os.path.dirname(feat_filename)
        if not os.path.exists(feat_dir):
            print 'Creting directory:', feat_dir
            os.mkdir(feat_dir)
        else:
            print 'Conv features directory exists:', feat_dir
        
        return svhn.generate_features_using_resnet_and_save(dir_path, count_to_process, feat_filename)

def get_data_for_end_end_train(features_file_name, bb_data_file_name, max_digits):
    # print 'Features File:', features_file_name
    # print 'bb_data File:', bb_data_file_name
    
    # svhn_data = svhn.load_pickle_file(bb_data_file_name)
    # pp.pprint(svhn_data[:10])    

    bb_data = svhn.get_svhn_digits_data(bb_data_file_name, max_digits)
    # pp.pprint(bb_data[:40])
    # bb_data = svhn.get_all_digits_bb_data(bb_data_file_name)
    test_features = svhn.load_resnet_features(features_file_name)

    # print 'bb_data Count:', len(bb_data), 'FeaturesCount:', len(test_features.keys())
    # print 'Features'
    # pp.pprint(test_features.keys()[:3])
    # print len(test_features)
    x = []
    # top = []
    # left = []
    # bottom = []
    # right = []
    # label = []

    # bb_data = bb_data[0:30]
    # pp.pprint(bb_data)
    # trn_bbox = np.stack([svhn.convert_bb(bbd, svhn.get_image_size(bbd[0])) for bbd in bb_data]).astype(np.float32)
    # lbs = np.stack(bbd[1].reshape(8) for bbd in bb_data)
    lbs = np.stack(bbd[1].reshape(max_digits + 1) for bbd in bb_data)
    file_names = [b[0] for b in bb_data]
    digits1 = [b[2][0].reshape(11) for b in bb_data]
    digits2 = [b[2][1].reshape(11) for b in bb_data]

    digits3 = []
    if (max_digits > 2):
        digits3 = [b[2][2].reshape(11) for b in bb_data]

    for bbd in bb_data:
        x.append(test_features[bbd[0]][1].reshape(1,2048))
    return (np.asarray(x), lbs, file_names, np.asarray(digits1), np.asarray(digits2), np.asarray(digits3))

def get_data_for_train(features_file_name, bb_data_file_name):
    print 'Features File:', features_file_name
    print 'bb_data File:', bb_data_file_name
    
    bb_data = svhn.get_all_digits_bb_data(bb_data_file_name)
    test_features = svhn.load_resnet_features(features_file_name)

    print 'bb_data Count:', len(bb_data), 'FeaturesCount:', len(test_features.keys())
    print 'Features'
    pp.pprint(test_features.keys()[:3])
    print len(test_features)
    x = []
    top = []
    left = []
    bottom = []
    right = []
    label = []

    # bb_data = bb_data[0:15]
    trn_bbox = np.stack([svhn.convert_bb(bbd, svhn.get_image_size(bbd[0])) for bbd in bb_data]).astype(np.float32)
    lbs = np.stack(np_utils.to_categorical(bbd[1][4].split('.')[0], 8).reshape(8) for bbd in bb_data)
    file_names = [b[0] for b in bb_data]
    

    for bbd in bb_data:
        x.append(test_features[bbd[0]][1].reshape(1,2048))
    return (np.asarray(x), trn_bbox, lbs, file_names)

def train(inp_shape, conv_feat, trn_bbox, trn_labels, conv_val_feat, val_bbox, val_labels, out_path):
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
    model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=1000, nb_epoch=1000, validation_data=(conv_val_feat, [val_bbox, val_labels]))
    model.save(out_path)

def train_end_end(inp_shape, conv_feat, trn_labels, digit1_labels, digit2_labels, digit3_labels, conv_val_feat, val_labels, val_digit1_labels, val_digit2_labels, val_digit3_labels, out_path, max_digits):
    print trn_labels.shape
    print digit1_labels.shape
    print digit2_labels.shape

    if os.path.exists(out_path):
        print 'Loading model from:', out_path
        model = load_model(out_path)
    else:
        inp = Input(inp_shape, name="input")
        x = Dense(1024, activation='relu')(inp)
        # x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Flatten()(x)
        
        # x_bb = Dense(4, name='bb')(x)
        x_class = Dense(max_digits + 1, activation='softmax', name='numDigits')(x)
        digit1 = Dense(11, activation='softmax', name='digit1')(x)
        digit2 = Dense(11, activation='softmax', name='digit2')(x)
        
        if (max_digits > 2):
            digit3 = Dense(11, activation='softmax', name='digit3')(x)
            model = Model([inp], [x_class, digit1, digit2, digit3])
            model.compile(Adam(lr=0.001), loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
        else:
            model = Model([inp], [x_class, digit1, digit2])
            model.compile(Adam(lr=0.001), loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    
    if max_digits > 2:
        model.fit(conv_feat, [trn_labels, digit1_labels, digit2_labels, digit3_labels], batch_size=1000, nb_epoch=50, validation_data=(conv_val_feat, [val_labels, val_digit1_labels, val_digit2_labels, val_digit3_labels]))
    else:
        model.fit(conv_feat, [trn_labels, digit1_labels, digit2_labels], batch_size=1000, nb_epoch=50, validation_data=(conv_val_feat, [val_labels, val_digit1_labels, val_digit2_labels]))
    print 'Saving model to:', out_path
    model.save(out_path)

def train_end_end_predictor(train_path, test_path, max_digits, model_output, skip_feature_gen=True, iterations=2):    
    print 'TrainPath:', train_path, 'TestPath:', test_path
    (train_svhn, train_feat_path) = get_data_files(train_path)
    (test_svhn, test_feat_path) = get_data_files(test_path)

    print "TrainFiles:", (train_svhn, train_feat_path)
    print "TestFiles:", (test_svhn, test_feat_path)

    if not skip_feature_gen:
        generate_conv_features_in_batches(train_path, train_feat_path, test_path, test_feat_path)

    
    (x_train, y_train_labels, train_files, digits1, digits2, digits3) = get_data_for_end_end_train(train_feat_path, train_svhn, max_digits=max_digits)
    
    (x_test, y_test_labels, test_files, test_digits1, test_digits2, test_digits3) = get_data_for_end_end_train(test_feat_path, test_svhn, max_digits=max_digits)

    print x_train[:3]
    print y_train_labels[:3]
    print train_files[:3]
    print digits1[:3]
    
    train_end_end((1, 2048), x_train, y_train_labels, digits1, digits2, digits3, x_test, y_test_labels, test_digits1, test_digits2, test_digits3, model_output, max_digits=max_digits)
    
def generate_conv_features_in_batches(train_path, train_feat_path, test_path, test_feat_path, end):
    test_done = False
    train_done = False
    for i in range(0, 220):
        if not test_done:
            test_done = generate_conv_features(test_path, test_feat_path)
        
        if not train_done:
            train_done = generate_conv_features(train_path, train_feat_path)
        
        if test_done and train_done:
            print "Test and training data generation done. Breaking"
            break

def train_bb_predictor(train_path, test_path, skip_feature_gen=True):    
    print 'TrainPath:', train_path, 'TestPath:', test_path
    (train_svhn, train_feat_path) = get_data_files(train_path)
    (test_svhn, test_feat_path) = get_data_files(test_path)

    print "TrainFiles:", (train_svhn, train_feat_path)
    print "TestFiles:", (test_svhn, test_feat_path)

    if not skip_feature_gen:
        generate_conv_features_in_batches(train_path, train_feat_path, test_path, test_feat_path)

    (x_train, y_train_bb, y_train_labels, train_files) = get_data_for_train(train_feat_path, train_svhn)
    (x_test, y_test_bb, y_test_labels, test_files) = get_data_for_train(test_feat_path, test_svhn)

    for i in range(0,2):
        train((1,2048), x_train, y_train_bb, y_train_labels, x_test, y_test_bb, y_test_labels, bb_predictor_model_output)
    
    return False

def train_digits_predictor_from_bb():
    return False

def predict_and_display(conv_feat, model_path, file_names):
    print 'Predicting'
    print conv_feat
    if os.path.exists(model_path):
        print 'Loading model from:', model_path
        model = load_model(model_path)
        pred = model.predict(conv_feat)
        
        for i, bb in enumerate(pred[0]):
            print i, bb
            print 'NumDigits:', np.argmax(pred[1][i])
            print 'NumDigits:', pred[1][i]
            svhn.plot_image_from_path(file_names[i], bb, target_size=(224,224))
            

def display_bb_predictions(data_path, model_path, count, generate_features=False):
    model_dir = os.path.join(data_path, 'models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    conv_feature_file_path = os.path.join(model_dir, conv_feature_file_name)

    if generate_features:
        generate_conv_features(data_path, conv_feature_file_path)

    conv_features = svhn.load_resnet_features(conv_feature_file_path)
    file_names = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')][:count]

    x = []
    for f in file_names:
        x.append(conv_features[f][1].reshape(1,2048))

    pp.pprint(file_names)
    predict_and_display(np.asarray(x), model_path, file_names)    

# train_bb_predictor(train_data_path, test_data_path, skip_feature_gen=False)

# display_bb_predictions(train_data_path, bb_predictor_model_output, 20)
# display_bb_predictions(test_data_path, bb_predictor_model_output, 5)
# display_bb_predictions(real_life_data_path, bb_predictor_model_output, 200, generate_features=True)

# train_digits_predictor_from_bb()

for i in range(0,100):
    train_end_end_predictor(train_data_path, test_data_path, max_digits=2, model_output = end_end_predictor_model_output_2_digits_no_drop_out, skip_feature_gen=True)
    train_end_end_predictor(train_data_path, test_data_path, max_digits=3, model_output = end_end_predictor_model_output_3_digits_no_drop_out, skip_feature_gen=True)


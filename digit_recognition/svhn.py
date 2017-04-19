import scipy.io as sio
import h5py
import pickle as pickle
import os
from PIL import Image
import pprint as pp
from operator import itemgetter
import numpy as np
from keras.utils import np_utils
import sys as sys
import pandas as pd
import time
import shutil

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from sets import Set
import random
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_reference(h5Ref, objectRef):
    return h5Ref[objectRef]

crop_digits_28_28_dirname = 'crop_digits_28_28'
resnet50_features_dirname = 'resnet50features'
resnet50_features_filename = 'resnet50_features.pkl'
#test_mat_file_name = '/Users/arun/ml/data/svhn/test_32x32.mat'
test_mat_file_name = '/Users/arunkumar/ml/data/svhn/test/digitStruct.mat'
test_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/test/digitStruct_parsed.pkl'
test_data_path = '/Users/arunkumar/ml/data/svhn/test'
test_28_28_digit_path = os.path.join(test_data_path, 'crop_digits_28_28')
test_bb_cropped_images = os.path.join(test_data_path, 'bb_cropped_images')

train_mat_file_name = '/Users/arunkumar/ml/data/svhn/train/digitStruct.mat'
train_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/train/digitStruct_parsed.pkl'
train_data_path = '/Users/arunkumar/ml/data/svhn/train'
train_28_28_digit_path = os.path.join(train_data_path, 'crop_digits_28_28')
train_bb_cropped_images = os.path.join(train_data_path, 'bb_cropped_images')

extra_train_mat_file_name = '/Users/arunkumar/ml/data/svhn/extra/digitStruct.mat'
extra_train_svhn_parsed_data_fileName = '/Users/arunkumar/ml/data/svhn/extra/digitStruct_parsed.pkl'
extra_data_path = '/Users/arunkumar/ml/data/svhn/extra'
extra_28_28_digit_path = os.path.join(extra_data_path, 'crop_digits_28_28')
extra_bb_cropped_images = os.path.join(extra_data_path, 'bb_cropped_images')

def get_image_files_in_dir(path):
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files if f.endswith('.png')]
    
    
def load_mat_file(file_name):
    print('Reading matlab file:', file_name)
    return h5py.File(file_name)

def get_file_name(ds):    
    return ''.join([chr(ds[i]) for i in range(0,len(ds))])

def parse_dataset1(ds, bb_data):            
    if ds.shape[0] == 1:
        return [ds[0][0]]
    else:        
        return [bb_data[ds[i][0]][0][0] for i in range(0, ds.shape[0])]

def get_bb_data(bb_data):
    refs = bb_data.get('#refs#')
    digitStruct = bb_data.get('digitStruct')    
    bbox = digitStruct['bbox']
    namesList = digitStruct['name']
    result = []
    print len(bbox)
    for i in range(0,len(bbox)):        
        item = bb_data[bbox[i][0]]

        labels = parse_dataset1(item.get('label'), bb_data)
        heights = parse_dataset1(item.get('height'), bb_data)
        lefts = parse_dataset1(item.get('left'), bb_data)
        tops = parse_dataset1(item.get('top'), bb_data)
        widths = parse_dataset1(item.get('width'), bb_data)

        img_name = get_file_name(bb_data[namesList[i][0]])
        digits = []
        for i in range(len(labels)):
            digits.append({'label':labels[i], 'left':lefts[i], 'top':tops[i], 'height':heights[i], 'width':widths[i]})

        result.append({'name':img_name, 'digits': digits})

    return result
    
def convert_from_mat_to_pickle(mat_file_name, pickle_file_name):        
    print 'Converting:', mat_file_name
    if os.path.exists(pickle_file_name):
        print 'SVHN data present at location:', pickle_file_name        
    else:
        bounding_box_data = load_mat_file(mat_file_name)
        svhn_data = get_bb_data(bounding_box_data)
        pickle.dump(svhn_data, open(pickle_file_name, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)

    #print svhn_data


def convert_svhn_data():
    convert_from_mat_to_pickle(test_mat_file_name, test_svhn_parsed_data_fileName)
    convert_from_mat_to_pickle(train_mat_file_name, train_svhn_parsed_data_fileName)
    convert_from_mat_to_pickle(extra_train_mat_file_name, extra_train_svhn_parsed_data_fileName)

def get_svhn_data_from_pickle(data_dir):
    files = os.listdir(data_dir)
    pickle_data = ''
    for f in files:
        if f.endswith('.pkl'):
            pickle_data = f
            break
    if not pickle_data:
        print 'No pickle file found:', pickle_data
        
    print 'Found pickle file:', pickle_data
    return pickle.load(open(os.path.join(data_dir, pickle_data), "rb"))
    

def crop_image_and_save(img, left, top, right, bottom, file_name, target_size=None):
    #print 'Writing to file', file_name
    if target_size:        
        img.crop((left, top, right, bottom)).resize(target_size).save(file_name)
    else:
        img.crop((left, top, right, bottom)).save(file_name)
    # img.crop((left, top, right, bottom)).resize((28,28)).save(crop_img_path)

def save_digits_from_training_sample(the_data, data_dir, crop_dir, target_size=(28,28)):
    img_path = os.path.join(data_dir, the_data['name'])        
    img = Image.open(img_path)    
    
    img_name_split = the_data['name'].split('.')
    for i,d in enumerate(the_data['digits']):    
        l = d['left']
        t = d['top']
        w = l + d['width']
        h = t + d['height']

        padding = 1
        l = max(0, l - padding)
        t = max(0, t - padding)
        w = min(img.size[0], w + padding)
        h = min(img.size[1], h + padding)

        crop_img_path = os.path.join(crop_dir, '{}.{}.{}.{}'.format(img_name_split[-2], i, int(d['label']), img_name_split[-1]))
        #print 'Saving to file:', crop_img_path
        img.crop((l, t, w, h)).save(crop_img_path)
        img.crop((l, t, w, h)).resize(target_size).save(crop_img_path)

def crop_digits_and_save(data_dir, crop_dir_name, target_size):
    svhn_data = get_svhn_data_from_pickle(data_dir)[:20]
    pp.pprint(svhn_data[:10])    
    
    crop_dir = os.path.join(data_dir, crop_dir_name)

    print 'Cropping digits from:', data_dir, 'Will be saved to:', crop_dir, 'targeSize:', target_size
    if not os.path.exists(crop_dir):
        print 'Creating directory:', crop_dir
        os.mkdir(crop_dir)
    else:
        print 'Crop dir', crop_dir, 'already present'

    for i in range (0, len(svhn_data)):
        save_digits_from_training_sample(svhn_data[i], data_dir, crop_dir, target_size=target_size)        
        if i % 1000 == 0:
            print i

def get_digit_lengths(data_dir):
    svhn_data = get_svhn_data_from_pickle(data_dir)
    digit_lengths = {}
    for i in range(0, len(svhn_data)):
        digits_len = len(svhn_data[i]['digits'])
        if not digit_lengths.has_key(digits_len):
            digit_lengths[digits_len] = 0
        digit_lengths[digits_len]+=1

    print digit_lengths

def get_image_size(img_path):
   im=Image.open(img_path)
   return (im.size[0], im.size[1])

def get_image_size_in_dir(data_dir):
    files = os.listdir(data_dir)    
    result = []    
    labels = ['name', 'width', 'height']
    for img_path in files:
        if img_path.endswith('.png'):
            fp = os.path.join(data_dir, img_path)
            im=Image.open(fp)
            result.append((fp, im.size[0], im.size[1]))
        else:
            print 'Skipping file:', img_path

    #pp.pprint(result)
    result = sorted(result, key=itemgetter(1))
    pp.pprint(result[:20])
    pp.pprint(result[-20:])

    df = pd.DataFrame.from_records(result, columns=labels)
    print df.describe()

def convert_image_to_array(path):
    img = Image.open(path)
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def get_png_files_in_dir(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    
def get_small_test_set(count):
    test_files = get_png_files_in_dir(test_28_28_digit_path)[:count]
    print len(test_files)

    test_data = np.asarray([convert_image_to_array(f) for f in test_files])
    test_labels = [f.split('.')[-2] for f in test_files]    
    test_y = np_utils.to_categorical(test_labels, 11)        
    
    test_data = test_data.astype('float32')
    test_data /= 255

    return (test_data, test_y ), (28,28, 3), test_files

def get_svhn_data():
    train_files = get_png_files_in_dir(train_28_28_digit_path) + get_png_files_in_dir(extra_28_28_digit_path)
    test_files = get_png_files_in_dir(test_28_28_digit_path)    

    train_data = np.asarray([convert_image_to_array(f) for f in train_files])
    train_labels = [f.split('.')[-2] for f in train_files]    
    train_y = np_utils.to_categorical(train_labels, 11)
    print train_labels[0:10]    
    print train_y[0:10]

    test_data = np.asarray([convert_image_to_array(f) for f in test_files])
    test_labels = [f.split('.')[-2] for f in test_files]    
    test_y = np_utils.to_categorical(test_labels, 11)        
    
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255
    test_data /= 255

    print len(train_data)
    print train_data[0].shape
    print test_data.shape
    print train_y.shape
    print test_y.shape    

    return (train_data, train_y), (test_data, test_y ), (28,28, 3)

def resize_and_get_svhn_data(svhn_label_file, target_size):
    
    svhn_data = get_svhn_digits_data(svhn_label_file, 3)
    # pp.pprint( svhn_data[:3])

    print 'Readling SVHN data from:', svhn_label_file, 'found:', len(svhn_data)    

    num_digits_data = []
    result = []
    for the_file in svhn_data:
        img = image.load_img(the_file[0], grayscale=False, target_size=target_size)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)        
        x = preprocess_input(x)
        result.append(x.reshape(x.shape[1],x.shape[2],x.shape[3]))

        num_digits_data.append(the_file[1].reshape(the_file[1].shape[1]))
        
        
    num_digits_data = np.stack(num_digits_data)
    return (np.asarray(result), num_digits_data)

def get_bb(one_image):
    left = sys.maxint
    top = sys.maxint
    bottom = 0
    right = 0
    d = one_image['digits']
    label = str(len(d)) + '.'
    for digit in d:
        l = digit['left']
        t = digit['top']
        h = digit['height']
        w = digit['width']
        bottom = max(bottom, t + h)
        right = max(right, l + w)
        label += '{}.'.format(int(digit['label']))
        left = min(left, l)
        top = min(top, t)
    return (top, left, bottom, right, label)

def get_all_digits_bb_data(bb_data_file):
    print 'Loading ', bb_data_file
    bb_data = pickle.load(open(bb_data_file, "rb"))
    # pp.pprint( bb_data[1000:1005])
    result = []
    for i in range (0, len(bb_data)):
        the_sample = bb_data[i]
        f_name = os.path.join(os.path.dirname(bb_data_file), the_sample['name'])
        result.append((f_name, get_bb(the_sample)))
    # pp.pprint(result[1000:1005])

    return result

def get_svhn_digits_data(svhn_file, max_digits):
    
    bb_data = load_pickle_file(svhn_file)
    
    # pp.pprint(bb_data[:41])
    result = []
    for i in range (0, len(bb_data)):
        the_sample = bb_data[i]
        f_name = os.path.join(os.path.dirname(svhn_file), the_sample['name'])
        digits = []
        
        sample_digits = the_sample['digits']
        for i in range(0, max_digits):
            if (i < len(sample_digits)):
                n = np_utils.to_categorical(sample_digits[i]['label'] % 10, 11)
            else:
                n = np_utils.to_categorical(10, 11)

            digits.append(n)
        
        # print the_sample
        digits_label = np_utils.to_categorical(min(max_digits, len(the_sample['digits'])), max_digits + 1)
        result.append((f_name, digits_label, np.asarray(digits)))
    # pp.pprint(result[1000:1005])

    return result


def save_bounded_box_image(data_dir, out_dir, target_size=None):
    print 'Cropping digits from:', data_dir, 'to', out_dir
    
    svhn_data = get_svhn_data_from_pickle(data_dir)
    pp.pprint(svhn_data[:10])    
    
    if not os.path.exists(out_dir):
        print 'Creating:', out_dir
        os.mkdir(out_dir)

    for i in range (0, len(svhn_data)):
        the_sample = svhn_data[i]
        (top, left, bottom, right, label) = get_bb(the_sample)
        img_path = os.path.join(data_dir, the_sample['name'])        
        img = Image.open(img_path)
        dest_file_name = os.path.join(out_dir, label + the_sample['name'])
        crop_image_and_save(img, left, top, right, bottom, dest_file_name, target_size)
    
        if i % 1000 == 0:
            print i    
    
def generate_features_using_resnet_and_save(data_path, count_to_process, pkl_results, target_size=(224, 224)):
    print 'Generating Resnet Convolution features for files in dir:', data_path
    print 'Features will be saved in:', pkl_results

    files = get_image_files_in_dir(data_path)
    random.shuffle(files)
    print len(files), 'files found'
    result = []
    model = ResNet50(weights='imagenet', include_top=False)
    # resnet_dest = os.path.join(data_path, resnet50_features_dirname)
    # pkl_results = os.path.join(resnet_dest, resnet50_features_filename)
    processed_set = Set()
    if os.path.exists(pkl_results):
        print 'Result file already exists. loading from', pkl_results        
        result = pickle.load(open(pkl_results, "rb"))
        for feature in result:
            processed_set.add(feature[0])
        
        if (len(files) == len(processed_set)):
            return True
    
    print time.ctime(), 'Processed:', len(processed_set)


    # print model.summary()
    skipped = 0
    predicted_count = 0
    for i, img_path in enumerate(files):
        # print img_path
        if img_path in processed_set:            
            skipped +=1
            continue

        img = image.load_img(img_path, grayscale=False, target_size=target_size)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predicted_count += 1
        preds = model.predict(x)
        result.append((img_path, preds))

        if result.count != 0 and i % 100 == 1:
            print x.shape
            print preds.shape
            print i, 'of', len(files), 'Skipped=', skipped, 'Predicted', predicted_count, 'result:', len(result), time.ctime()
            
            
            # if  not os.path.exists(resnet_dest):
            #     print 'Creating directory:', resnet_dest
            #     os.makedirs(resnet_dest)

            
            # print 'Writing to file:', pkl_results, 'count:', len(result)
            
            # pickle.dump(result, open(pkl_results, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
        if predicted_count > count_to_process:            
            print 'Processed', predicted_count, 'Breaking'
            break
            
    print 'Writing to file:', pkl_results, 'count:', len(result)
    pickle.dump(result, open(pkl_results, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])

    return False

def load_pickle_file(path):
    print 'Loading pickle file', path
    return pickle.load(open(path, "rb"))

def get_resnt50_features_data(path):
    train_results_path = os.path.join(os.path.join(path, resnet50_features_dirname), resnet50_features_filename)
    return pickle.load(open(train_results_path, "rb"))

def generate_resnet50_for_path(path, file_name):
    print('Generating resnet50 features. Path:', path, 'Output:', file_name)

def to_plot(img):
    return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

def create_rect(bb, color='red'):
    print 'Rectange bb:', bb
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color, fill=False, lw=3)

# def create_rect(bb, color='red'):
#     return plt.Rectangle((5,5), 5, 5, color=color, fill=False, lw=3)

def plot_image_from_path(path, bb, target_size = None):
    print 'Plotting image from path:', path
    print 'Image size:', get_image_size(path)
    # img=mpimg.imread(path)

    # img = image.load_img(path, grayscale=False, target_size=(224,224))
    img = image.load_img(path, grayscale=False, target_size=target_size)
    # x = image.img_to_array(img)

    imgplot = plt.imshow(img)
    plt.gca().add_patch(create_rect(bb))
    plt.show()

    
def move_random_files(from_dir, to_dir, count, tag):
    print 'Moving ', count, 'file from', from_dir, 'to', to_dir
    src_beforeCount = len(os.listdir(from_dir))
    dest_beforeCount = len(os.listdir(to_dir))
    files = [f for f in os.listdir(from_dir) if f.endswith('.png')]
    files_to_move = [files[i] for i in np.random.choice(len(files), count)]
    for f in files_to_move:
        src = os.path.join(from_dir, f)
        des = os.path.join(to_dir, tag + f)
        print 'Moving file:', src, 'to', des
        shutil.move(src, des)
    
    print 'Source: Before', src_beforeCount, 'After:', len(os.listdir(from_dir))
    print 'Dest: Before', dest_beforeCount, 'DestAfter:', len(os.listdir(to_dir))
    

def move_files_and_rename(from_dir, to_dir):
    files = [f for f in os.listdir(from_dir) if f.endswith('.png')]
    for f in files:
        src = os.path.join(from_dir, f)
        des = os.path.join(to_dir, '.'.join(f.split('.')[1:]))
        print 'Moving file:', src, 'to', des
        shutil.move(src, des)    

def load_resnet_features(file_name):
    result = {}    
    for item in load_pickle_file(file_name):
        result[item[0]] = item
    return result
    
def convert_bb(bb_data, size):
    bb = []
    f = bb_data[1]
    bb.append(f[0])
    bb.append(f[1])
    bb.append(f[2])
    bb.append(f[3])    
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])    
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_y, 0)
    bb[3] = max(bb[3]*conv_x, 0)    
    return bb

#convert_svhn_data()
# crop_digits_and_save(test_data_path)
# crop_digits_and_save(train_data_path)
# crop_digits_and_save(extra_data_path)
# get_digit_lengths(test_data_path)
# get_digit_lengths(train_data_path)
# get_digit_lengths(extra_data_path)

# get_cropped_image_size(train_bb_cropped_images)

# convert_image_to_array('/Users/arunkumar/ml/data/svhn/test/crop_digits_28_28/9989.1.3.png')

# print len(get_png_files_in_dir(test_data_path))
# print len(get_png_files_in_dir(train_data_path))

# get_svhn_data()


# save_bounded_box_image(train_data_path, train_bb_cropped_images)
# save_bounded_box_image(test_data_path, test_bb_cropped_images)
# save_bounded_box_image(extra_data_path, extra_bb_cropped_images)

# for i in range(0, 300):
#     # generate_features_using_resnet_and_save(test_bb_cropped_images, 1002)
#     # generate_features_using_resnet_and_save(train_bb_cropped_images, 1002)
#     generate_features_using_resnet_and_save(extra_bb_cropped_images, 1000)
    
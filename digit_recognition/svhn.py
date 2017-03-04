import scipy.io as sio
import h5py
import pickle as pickle
import os

def get_reference(h5Ref, objectRef):
    return h5Ref[objectRef]

#test_mat_file_name = '/Users/arun/ml/data/svhn/test_32x32.mat'
test_mat_file_name = '/Users/arun/ml/data/svhn/test/digitStruct.mat'
svhn_parsed_data_fileName = '/Users/arun/ml/data/svhn/test/digitStruct_parsed.pkl'

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

        result.append((img_name, labels, heights, lefts, tops, widths))

    return result
    

if os.path.exists(svhn_parsed_data_fileName):
    print 'Loading SVHN data from:', svhn_parsed_data_fileName
    svhn_data = pickle.load(open(svhn_parsed_data_fileName, "rb"))
else:
    bounding_box_data = load_mat_file(test_mat_file_name)
    svhn_data = get_bb_data(bounding_box_data)
    pickle.dump(svhn_data, open(svhn_parsed_data_fileName, "wb" ))

print svhn_data



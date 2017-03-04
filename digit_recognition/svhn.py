import scipy.io as sio
import h5py

def get_reference(h5Ref, objectRef):
    return h5Ref[objectRef]

#test_mat_file_name = '/Users/arun/ml/data/svhn/test_32x32.mat'
test_mat_file_name = '/Users/arun/ml/data/svhn/test/digitStruct.mat'

def load_mat_file(file_name):
    print('Reading matlab file:', file_name)
    return h5py.File(file_name)

def get_file_name(ds):    
    return ''.join([chr(ds[i]) for i in range(0,len(ds))])

def parse_dataset1(ds, bb_data):    
    index = 0
    # print 'ds:', ds
    # print 'ds.shape:', ds.shape
    # print 'ds:', type(ds)
    # print 'ds[0]', ds[0]
    # print 'ds[0][0]', ds[index][0]
    if ds.shape[0] == 1:
        return [ds[0][0]]
    else:
        # print 'ds[1][0]', ds[1][0]
        # print 'bb_data[ds[1][0]]', bb_data[ds[1][0]]
        # print 'bb_data[ds[1][0]][0][0]', bb_data[ds[1][0]][0][0]
        return [bb_data[ds[i][0]][0][0] for i in range(0, ds.shape[0])]

# def get_image_names(bb_data):    
#     digitStruct = bb_data.get('digitStruct')
#     print "digitStruct", digitStruct
    
#     namesList = digitStruct['name']
#     print 'namelist:', namesList
#     for i in range(0,10):
#         print parse_dataset(bb_data[namesList[i][0]])

def get_bb_data(bb_data):
    refs = bb_data.get('#refs#')
    digitStruct = bb_data.get('digitStruct')    
    bbox = digitStruct['bbox']
    namesList = digitStruct['name']
    result = []
    for i in range(0,2):
        print '******************  ', i, '  ******************'
        labels = parse_dataset1(bb_data[bbox[i][0]].get('label'), bb_data)
        heights = parse_dataset1(bb_data[bbox[i][0]].get('height'), bb_data)
        lefts = parse_dataset1(bb_data[bbox[i][0]].get('left'), bb_data)
        tops = parse_dataset1(bb_data[bbox[i][0]].get('top'), bb_data)
        widths = parse_dataset1(bb_data[bbox[i][0]].get('width'), bb_data)
        img_name = get_file_name(bb_data[namesList[i][0]])
        result.append((img_name, labels, heights, lefts, tops, widths))
    return result
    


bounding_box_data = load_mat_file(test_mat_file_name)
# print bounding_box_data
# print bounding_box_data.keys()
# for k, v in bounding_box_data.iteritems():
#     print k, len(v)

# get_image_names(bounding_box_data)
print get_bb_data(bounding_box_data)



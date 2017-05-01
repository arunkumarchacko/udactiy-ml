from PIL import Image
from operator import itemgetter
import os
import pprint as pp


filepath = '/Users/arun/ml/data/svhn/test/'

files = os.listdir(filepath)
print files[0:5]
result = []
a = ''
for img_path in files:
    if img_path.endswith('.png'):
        fp = os.path.join(filepath, img_path)
        im=Image.open(fp)
        result.append((fp, im.size[0], im.size[1]))
    else:
        print 'Skipping file:', img_path

#pp.pprint(result)
result = sorted(result, key=itemgetter(1))

pp.pprint(result[0:25])
pp.pprint(result[-25:])

im = Image.open(result[0][0])
im.show()

im = Image.open(result[10][0])
im.show()

im = Image.open(result[20][0])
im.show()

im = Image.open(result[-1][0])
im.show()

im = Image.open(result[-2][0])
im.show()
#print len(result)
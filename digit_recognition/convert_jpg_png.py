import os
from PIL import Image

dir = '/Users/arunkumar/ml/data/svhn/internet/'
files = os.listdir(dir)
for f in files:
    print f
    im = Image.open(os.path.join(dir, f))
    im.save(os.path.join(dir, f + '.png'))
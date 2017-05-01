import svhn
from PIL import Image
from keras.preprocessing import image

from matplotlib import pyplot as plt

fp = '/Users/arunkumar/ml/data/svhn/train/16684.png'
# im=Image.open(fp)

# svhn.plot(im)

def create_rect(bb, color='red'):
    return plt.Rectangle((5,5), 5, 5, color=color, fill=False, lw=3)

def show_bb(i):
    # bb = val_bbox[i]
    # plot(val[i])
    plt.gca().add_patch(create_rect(''))

print svhn.get_image_size(fp)
svhn.plot_image_from_path(fp)


from PIL import Image
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import numpy as np
import pylab


def load_image(path):
    im = spnd.imread(path, mode='L')
    im = np.array(im, np.float32)
    im = im / 255
    return im


def load_image_origin(path):
    im = spnd.imread(path)
    im = np.array(im, np.float32)
    im = im / 255
    return im


def plot_image(ims=[], title=[], col=1):
    if len(ims) < 1:
        return
    num = len(ims)
    f = False
    if len(title) == len(ims):
        f = True
    if num % col != 0:
        max_row = (num + 1) / col
    else:
        max_row = num / col
    fig = plt.figure()
    for i in xrange(0, num):
        t = fig.add_subplot(col, max_row, i + 1)
        if f:
            t.set_title(title[i])
        plt.imshow(ims[i], cmap='gray')
    plt.show()
    return

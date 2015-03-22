import scipy.ndimage as spnd
import scipy.misc as misc
import numpy as np
import pylab
from util import load_image, plot_image


p = 'D:\\Dropbox\\Computer Vision\\code\\c3\\data.png'
p2 = 'D:\\Dropbox\\Computer Vision\\code\\c3\\data2.jpg'


def cv1_test_function():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x) / x
    pylab.plot(x, y)
    pylab.show()


def cv2_test_interpolate():
    im = load_image(p)
    out1 = misc.imresize(im, 1000, interp='nearest')
    out2 = misc.imresize(im, 1000, interp='bilinear')
    out3 = misc.imresize(im, 1000, interp='bicubic')
    plot_image(ims=[out1, out2, out3], title=[
               'nearest', 'bilinear', 'bicubic'], col=2)


def _sub_sampling(img, s=2):
    out1 = []
    for index, row in enumerate(img):
        if index % s != 0:
            continue
        tem = []
        for j, elem in enumerate(row):
            if j % s != 0:
                continue
            tem.append(elem)
        out1.append(tem)
    return np.array(out1, np.float32)


def _gau_filter(img):
    sigma = 1
    x, y = len(img), len(img[0])
    out1 = np.zeros((x, y))
    spnd.filters.gaussian_filter(input=img, sigma=sigma, output=out1)
    return out1


def cv2_test_subsampling():
    im = load_image(p2)
    out1 = _sub_sampling(im, s=5)
    gau = _gau_filter(im)
    out2 = _sub_sampling(gau, s=5)
    plot_image(ims=[im, out1, out2], title=[
               'origin', 'subsampling', 'gaussian + subsampling'], col=2)

# cv1_test_function()
cv2_test_interpolate()
# cv2_test_subsampling()

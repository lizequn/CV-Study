import scipy.ndimage as spnd
import numpy as np
import pylab
from util import load_image, plot_image

p = 'D:\\Dropbox\\Computer Vision\\code\\c1\\data.png'


def cv1_test_1(img):
    r = 100. / 255
    result1 = []
    for el1 in img:
        result1.append(map(lambda x: x + r if x + r <= 1 else 1, el1))
    result2 = []
    for el1 in img:
        result2.append([i for i in reversed(el1)])
    return [result1, result2]


def cv1_test_cov(img):
    x, y = len(img), len(img[0])
    out1 = np.zeros((x, y))
    w1 = [[1. / 9, 1. / 9, 1. / 9],
          [1. / 9, 1. / 9, 1. / 9], [1. / 9, 1. / 9, 1. / 9]]
    spnd.filters.convolve(input=img, weights=w1, output=out1)
    out2 = np.zeros((x, y))
    w2 = [[-1. / 9, -1. / 9, -1. / 9],
          [-1. / 9, 2 - 1. / 9, -1. / 9], [-1. / 9, -1. / 9, -1. / 9]]
    spnd.filters.convolve(input=img, weights=w2, output=out2)
    return [out1, out2]


def cv1_test_gaussian():
    sigma = 3
    x = np.linspace(-10, 10, 100)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- x**2 / (2 * sigma**2))
    pylab.plot(x, y)
    pylab.show()


def cv1_test_gaufilter(img, sigma=3):
    x, y = len(img), len(img[0])
    out1 = np.zeros((x, y))
    spnd.filters.gaussian_filter(input=img, sigma=sigma, output=out1)
    return out1

im = load_image(p)
rm = cv1_test_1(im)
plot_image(ims=[im, rm[0], rm[1]], col=2)

im = load_image(p)
rm = cv1_test_cov(im)
plot_image(ims=[im, rm[0], rm[1]], title=[
           'Origin', 'Blur', 'Sharpening'], col=2)

cv1_test_gaussian()

im = load_image(p)
rm2 = cv1_test_gaufilter(im)
rm1 = cv1_test_gaufilter(im, 1)
rm3 = cv1_test_gaufilter(im, 5)
plot_image(ims=[im, rm1, rm2, rm3], title=[
           'Origin', '$\sigma = 1$', '$\sigma = 3$', '$\sigma = 5$'], col=2)

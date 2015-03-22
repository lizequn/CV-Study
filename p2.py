import scipy.ndimage as spnd
import numpy as np
from util import load_image, plot_image

p = 'D:\\Dropbox\\Computer Vision\\code\\c2\\data.png'


def cv2_sobel(img):
    x, y = len(img), len(img[0])
    out1 = np.zeros((x, y))
    spnd.filters.sobel(input=img, output=out1, axis=0)
    out2 = np.zeros((x, y))
    spnd.filters.sobel(input=img, output=out2, axis=1)
    out3 = np.sqrt(out1**2 + out2**2)
    return (out1, out2, out3)

im = load_image(p)
rm1, rm2, rm3 = cv2_sobel(im)
plot_image(ims=[im, rm1, rm2, rm3], title=[
           'origin', 'sobel x', 'sobel y', 'gradient magnitude'], col=2)

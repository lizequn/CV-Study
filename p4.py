import scipy.ndimage as spnd
import numpy as np
from util import load_image
import pylab

# todo un
p = 'D:\\Dropbox\\Computer Vision\\code\\c4\\data.png'


def test_har_detect(im, sigma=1):
    min_dist = 10
    pert = 0.1
    imx = np.zeros(im.shape)
    spnd.filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    spnd.filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    Ixx = spnd.filters.gaussian_filter(imx * imx, sigma)
    Iyy = spnd.filters.gaussian_filter(imy * imy, sigma)
    Ixy = spnd.filters.gaussian_filter(imx * imy, sigma)
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    h_mat = det / trace
    thre = h_mat.max() * pert
    h_t = (h_mat > thre) * 1
    p_mat = np.array(h_t.nonzero()).T
    re_mat = [h_mat[c[0], c[1]] for c in p_mat]
    index = np.argsort(re_mat)[::-1]
    allowed_locations = np.zeros(h_mat.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    result = []
    for i in index:
        if allowed_locations[p_mat[i, 0], p_mat[i, 1]] == 1:
            result.append(p_mat[i])
            allowed_locations[
                (p_mat[i, 0] - min_dist):(p_mat[i, 0] + min_dist),
                (p_mat[i, 1] - min_dist):(p_mat[i, 1] + min_dist)] = 0
    pylab.figure()
    pylab.gray()
    pylab.imshow(im)
    pylab.plot([p[1] for p in result],
               [p[0] for p in result], 'o')
    pylab.axis('off')
    pylab.show()

im = load_image(p)
test_har_detect(im)

from PIL import Image
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import numpy as np
import pylab

p = 'D:\\Dropbox\\Computer Vision\\code\\c1\\data.png'
def load_image(path):
    im = spnd.imread(path,mode='L')
    im = np.array(im,np.float32)
    im = im/255
    return im

def plot_image(ims=[],title = [],col=1):
    if len(ims)<1:
        return
    num = len(ims)
    f = False
    if len(title)== len(ims):
        f = True
    if num%col != 0:
        max_row = (num+1)/col
    else:
        max_row = num/col
    fig = plt.figure()
    for i in xrange(0,num):
        t = fig.add_subplot(col,max_row,i+1)
        if f:
            t.set_title(title[i])
        plt.imshow(ims[i],cmap='gray')
    plt.show()
    return

def cv1_test_1(img):
    r = 100./255
    result1 = []
    for el1 in img:
        result1.append(map(lambda x: x+r if x+r<=1 else 1,el1))
    result2 = []
    for el1 in img:
        result2.append([i for i in reversed(el1)])
    return [result1,result2]
def cv1_test_cov(img):
    x,y = len(img),len(img[0])
    out1 = np.zeros((x,y))
    w1  = [[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
    spnd.filters.convolve(input = img,weights = w1,output=out1)
    out2 = np.zeros((x,y))
    w2  = [[-1./9,-1./9,-1./9],[-1./9,2-1./9,-1./9],[-1./9,-1./9,-1./9]]
    spnd.filters.convolve(input = img,weights = w2,output=out2)
    return [out1,out2]
def cv1_test_gaussian():
    sigma = 3
    x = np.linspace(-10,10,100)
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- x**2 / (2 * sigma**2))
    pylab.plot(x,y)
    pylab.show()


im = load_image(p)
rm = cv1_test_1(im)
plot_image(ims = [im,rm[0],rm[1]],col=2)

im = load_image(p)
rm = cv1_test_cov(im)
plot_image(ims = [im,rm[0],rm[1]],title = ['Origin','Blur','Sharpening'],col=2)

cv1_test_gaussian()

from PIL import Image
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import numpy as np
import pylab

p = 'D:\\Dropbox\\Computer Vision\\code\\c2\\data.png'
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

def cv2_sobel(img):
    x,y = len(img),len(img[0])
    out1 = np.zeros((x,y))
    spnd.filters.sobel(input = img,output = out1,axis = 0)
    out2 = np.zeros((x,y))
    spnd.filters.sobel(input = img,output = out2,axis = 1)
    out3 = np.sqrt(out1**2 + out2**2)
    return (out1,out2,out3)





im = load_image(p)
rm1,rm2,rm3 = cv2_sobel(im)
plot_image(ims = [im,rm1,rm2,rm3],title = ['origin','sobel x','sobel y','gradient magnitude'],col = 2)

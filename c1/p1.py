from PIL import Image
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import numpy as np

p = 'D:\\Dropbox\\Computer Vision\\c1\\data.png'
H = [[0,0,0],[1,0,0],[0,0,0]]
def load_image(path):
    im = spnd.imread(path,mode='L')
    im = np.array(im,np.float32)
    im = im/255
    return im

def plot_image(ims=[],title = [],col=1):
    if len(ims)<1:
        return
    num = len(ims)
    if num%col != 0:
        max_row = (num+1)/col
    else:
        max_row = num/col
    fig = plt.figure()
    for i in xrange(0,num):
        fig.add_subplot(col,max_row,i+1)
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

im = load_image(p)
rm = cv1_test_1(im)
plot_image(ims = [im,rm[0],rm[1]],col=2)

def conv_filter(F,H):
    k = len(H)
    if k < 3:
        return
    if k!=len(H[0]):
        return
    off = (k-1)/2
    mi = len(F)
    mj = len(F[0])
    G = np.zeros((mi,mj))
    for i in xrange(0,mi):
        for j in xrange(0,mj):
            sum = 0
            for u in xrange(0,k+1):
                for v in xrange(0,k+1):
                    ru = u-off
                    rv = v-off
                    if i-ru < 0 or i-ru >=mi:
                        continue
                    if j-rv < 0 or j-rv >=mj:
                        continue
                    sum += F[i-ru][j-rv] * H[u][v]
            G[i][j] = sum
    G = np.array(G,np.float32)
    return G

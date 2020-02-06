import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 10) % self.slices
        else:
            self.ind = (self.ind - 10) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

if __name__ == "__main__":
    #How to use:
    for i in tqdm(range(1800,1900)):
        x = cv2.imread('../../Data/uCT/EK_208_215/EK_208_215_'+(str(i).zfill(4))+'.tif')
        color.append(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        #x = cv2.GaussianBlur(x, (5,5), cv2.BORDER_DEFAULT)
        ret, x = cv2.threshold(x, 50, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        ct.append(x)
    ct = np.array(ct)

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, ct.T)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    save = str(input('Save figure? '))
    if save:
        fig.savefig('output/'+save+'.png')

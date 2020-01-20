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

def findrows(df, col, value):
    #Find all rows that have specified value in specified column
    #e.g. find all rows that have 12 in column 'age'
    return list(df.loc[df[col]==value].index.values)

def trim(df, col, value):
    #Trim df to e.g. fish that are 12 years old
    index = findrows(df, col, value)
    trimmed = df.drop(set(df.index) - set(index))
    return trimmed

def crop(ct, circles):
    CTs = []
    for x, y, r in circles:
        c = []
        for slice_ in ct:
            rectX = (x - r) 
            rectY = (y - r)
            cropped_slice = slice_[rectY:(y+2*r), rectX:(x+2*r)]
            c.append(cropped_slice)
        CTs.append(c)

    return CTs

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

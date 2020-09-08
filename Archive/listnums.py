import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()

master = ctreader.mastersheet()


# Group all ages into 6, 12, 24, 36 months old
master.age = pd.cut(master.age, bins = 4, labels=[6, 12, 24, 36], right = False)

age6    = ctreader.trim(master, 'age', 6)
age12   = ctreader.trim(master, 'age', 12)
age24   = ctreader.trim(master, 'age', 24)
age36   = ctreader.trim(master, 'age', 36)

num6    = [age6['genotype'].value_counts()['wt'], age6['genotype'].value_counts()['het'], age6['genotype'].value_counts()['hom']]
num12   = [age12['genotype'].value_counts()['wt'], age12['genotype'].value_counts()['het'], age12['genotype'].value_counts()['hom']]
num24   = [age24['genotype'].value_counts()['wt'], age24['genotype'].value_counts()['het'], age24['genotype'].value_counts()['hom']]
num36   = [age36['genotype'].value_counts()['wt'], age36['genotype'].value_counts()['het'], age36['genotype'].value_counts()['hom']]

# W, H, H
data = np.array([
[num6[0] , num6[1], num6[2]], # 6
[num12[0] , num12[1], num12[2]], # 12
[num24[0] , num24[1], num24[2]], # 24
[num36[0] , num36[1], num36[2]], # 36
])
print(data)

column_names = [None, 'wt', None, 'het', None, 'hom']
row_names = [ None, '6', None, '12', None, '24', None, '36']

fig = plt.figure()
ax = Axes3D(fig)

lx= len(data[0])            # Work out matrix dimensions
ly= len(data[:,0])
xpos = np.arange(0,lx,1)    # Set up a mesh of positions
ypos = np.arange(0,ly,1)
xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

xpos = xpos.flatten()   # Convert positions to 1D array
ypos = ypos.flatten()
zpos = np.zeros(lx*ly)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = data.flatten()

cs = ['r', 'g', 'b'] * ly

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)

ax.tick_params(axis='both', which='major', labelsize=14)

ax.w_xaxis.set_ticklabels(column_names)
# for tick in ax.w_xaxis.get_major_ticks():
#     tick.label.set_fontsize(12)
ax.w_yaxis.set_ticklabels(row_names)
# for tick in ax.w_yaxis.get_major_ticks():
#     tick.label.set_fontsize(12)

ax.set_xlabel('Genotype', fontsize=20, labelpad=20)
ax.set_ylabel('Age', fontsize=20, labelpad=20)
ax.set_zlabel('Occurrence', fontsize=20, labelpad=20)



plt.show()
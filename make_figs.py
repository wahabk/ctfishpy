import ctfishpy
import numpy as np
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()

z, y, x = ctreader.read_max_projections(40)
label = ctreader.read_label('Otoliths', 40)
zt, zy, zx = ctreader.make_max_projections(label)

print(y.shape)
y[zy != 0] = [255,0,0]
plt.imshow(y)
plt.show()

import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet()
label = unet.predict(40)
print(label.shape, np.max(label), np.unique(label))
ctreader.write_label(label, 'Otoliths-unet', 40)
plt.imsave('output/Images/second_prediction2.png', label[120])
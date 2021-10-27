import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
from scipy.interpolate import interpn
from scipy import stats

from numpy import cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def plot_pdf(x):
	bins = np.linspace(-500, 20000, 3000)
	bin_centers = 0.5*(bins[1:] + bins[:-1])
	pdf = stats.norm.pdf(bin_centers)

	samples = x.flatten()
	print(x.shape, samples.shape, samples[0])
	histogram, bins = np.histogram(samples, bins=bins, density=True)

	plt.figure(figsize=(6, 4))
	plt.plot(bin_centers, histogram)
	# plt.plot(bin_centers, pdf, label="PDF")
	plt.show()

def gauss(x, a, b, c):
	return a*np.exp( -(x - b)**2/c )

def transferFunction(x):
	gauss(x, 1.0, 9.0, 1.0)
	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
	
	return r,g,b,a


if __name__=='__main__':
	n=40
	organ = 'Otoliths'
	segs = 'Otoliths_unet2d'
	roiSize = (128,128,128)
	is_amira = False

	ctreader = ctfishpy.CTreader()
	center = ctreader.manual_centers[str(n)]

	ct, stack_metadata = ctreader.read(n, align=True)
	ct = ctreader.crop3d(ct, roiSize, center=center)
	# label = ctreader.read_label(segs, n, is_amira=is_amira)
	# label = ctreader.crop3d(label, roiSize, center=center)
	# ct = ct / ct.max()

	""" Volume Rendering """
	a, b, c = ct.shape

	X, Y, Z = np.mgrid[-8:8:128j, -8:8:128j, -8:8:128j]
	values = np.sin(X*Y*Z) / (X*Y*Z)
	# values = ((ct - ct.mean()) / ct.max())*20
	values=ct
	print(values.shape, values.min(), values.max())
	print(X.shape, X.min(), X.max())

	fig = go.Figure(data=go.Volume(
		x=X.flatten(),
		y=Y.flatten(),
		z=Z.flatten(),
		value=values.flatten(),
		isomin=15000,
		# isomax=50000,
		opacity=0.8, # needs to be small to see through all surfaces
		surface_count=5, # needs to be a large number for good volume rendering
		# opacityscale=[[500, 1], [2000, 0], [3000, 0.25], [12000, 1]],
		opacityscale='max',
		))
	fig.show()

	# # Load Datacube
	# plot_pdf(ct)

	# datacube = ct
	# print(datacube.shape, datacube.min(), datacube.max())
	
	# # Datacube Grid
	# Nx, Ny, Nz = datacube.shape
	# x = np.linspace(-Nx/2, Nx/2, Nx)
	# y = np.linspace(-Ny/2, Ny/2, Ny)
	# z = np.linspace(-Nz/2, Nz/2, Nz)
	# points = (x, y, z)
	
	# # Do Volume Rendering at Different Veiwing Angles
	# Nangles = 1
	# for i in range(Nangles):
		
	# 	print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
	
	# 	# Camera Grid / Query Points -- rotate camera view
	# 	angle = np.pi/2 * i / Nangles
	# 	N = 180
	# 	c = np.linspace(-N/2, N/2, N)
	# 	qx, qy, qz = np.meshgrid(c,c,c)
	# 	qxR = qx
	# 	qyR = qy * np.cos(angle) - qz * np.sin(angle) 
	# 	qzR = qy * np.sin(angle) + qz * np.cos(angle)
	# 	qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
		
	# 	# Interpolate onto Camera Grid
	# 	camera_grid = interpn(points, datacube, qi, method='linear')
	# 	camera_grid = camera_grid.reshape((N,N,N))
		
	# 	# Do Volume Rendering
	# 	image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
	
	# 	for dataslice in camera_grid:
	# 		r,g,b,a = transferFunction(np.log(dataslice))
	# 		image[:,:,0] = a*r + (1-a)*image[:,:,0]
	# 		image[:,:,1] = a*g + (1-a)*image[:,:,1]
	# 		image[:,:,2] = a*b + (1-a)*image[:,:,2]
		
	# 	image = np.clip(image,0.0,1.0)
		
	# 	# Plot Volume Rendering
	# 	plt.figure(figsize=(4,4), dpi=80)
		
	# 	plt.imshow(image)
	# 	plt.axis('off')
		
	# 	# Save figure	
	
	
	# # Plot Simple Projection -- for Comparison
	# plt.figure(figsize=(4,4), dpi=80)
	
	# plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
	# plt.clim(-5, 5)
	# plt.axis('off')
	
	# # Save figure
	# # plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
	# plt.show()
	
	
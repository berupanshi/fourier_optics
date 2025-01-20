"""

PSF and Tranfer Function

"""

from IPython import get_ipython; 
get_ipython().run_line_magic('reset','-sf')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2 

#%% Defining 2D space

p = 3 #  pixel size (in micrometre)

dx = p # lateral sample interval/pixel size
dy = p

N = 512 # N*N computational window

Lx = N*dx # Sampling period
Ly = N*dy 

m, n = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2)) 

x = m*dx # to take it to real dimension
y = n*dy


#%% Original Object for our simulation study

img = cv2.imread("Downloads/Lena Image.png",0)

#%% PSF

a = 20
gaussian2d = np.exp(-(x**2+y**2)/(a**2))

conv_result = convolve2d(img, gaussian2d, mode='same')
#%%
# Plotting the result
plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Object")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(gaussian2d, cmap='gray')
plt.title("PSF of Imaging System")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(conv_result, cmap='gray')
plt.title("Output Image")
plt.colorbar()

plt.show()

#%% Transfer Function

ft_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
ft_gaussian = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian2d)))

ft_output = ft_img*ft_gaussian

output_image= np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ft_output)))

#%%
# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

plt.imshow(abs(output_image)), plt.colorbar()
plt.title("Result by PSF Method")

plt.subplot(1, 2, 2)
plt.imshow(conv_result), plt.colorbar()
plt.title("Result by Transfer Function Method")

plt.show()


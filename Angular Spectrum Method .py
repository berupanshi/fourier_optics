"""

Fresnel Diffraction using Angular Spectrum Method (ASM)


"""

from IPython import get_ipython; 
get_ipython().run_line_magic('reset','-sf')


import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
import cv2

#%%


p = 3e-6 #  pixel size (in micrometre)

dx = p # lateral sample interval/pixel size
dy = p

N = 200 # N*N computational window

Lx = N*dx # Sampling period
Ly = N*dy 


# Real Space grid
m, n = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2)) 

x = m*dx # to take it to real/physical dimension
y = n*dy

# Frequency grid

fx, fy = np.meshgrid(np.arange(-0.5, 0.5, 1/N), np.arange(-0.5, 0.5, 1/N))
fx = fx/dx
fy = fy/dy

# Wavelength and propagation constant

ld = 500e-9
k = 2*np.pi/ld


#%%

# Define input plane/ transmittance function/ aperture function

wx = 20*dx # half width i.e. rect spread over 20 samples
wy = 20*dx 

rect = np.where((np.abs(x) <= wx) * (np.abs(y) <= wy), 1.0, 0.0)
 
inp = rect




#%%

# Define Input field

u1 = 1*inp

# Define Angular Spectrum Transfer function

z = 1000e-6 # Try different distances and different sizes of rect

# Define alpha matrix

alpha = np.sqrt(k**2 - 4*np.pi**2*(fx**2 + fy**2))


H = np.exp(1j*alpha*z)



# Do the Fourier transform of input plane

U1 = fftshift(fft2(ifftshift(u1)))

# Fourier transform of output field using ASM

U2 = U1 * H

# Finally, output field = Fresnel diffraction of input field

u2 = fftshift(ifft2(ifftshift(U2)))



# Plotting the result

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(u1)
plt.title("u1: Input Field")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.abs(U1)) # Try angle
plt.title("U1: Fourier transform of u1") 
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(np.abs(U2)) # Try angle
plt.title("U2: Fourier transform of u2")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(abs(u2)**2)
plt.title("u2: Output field/Fresnel diffraction pattern")
plt.colorbar()

plt.show()




#%% Try these aperture functions


# Double-delta function

double_delta = np.zeros([200,200], dtype='float')
double_delta[90,90] = 1
double_delta[90,110] = 1 

inp = double_delta

# Your Image

img = plt.imread('Squares.png')
img = cv2.resize(img, [200,200])

inp = img
#%%
# Gaussian function

a = 100e-6
gaussian2d = np.exp(-(x**2+y**2)/(a**2))

inp = gaussian2d

#%%
#%% Original Object for our simulation study

img = cv2.imread("Downloads/Lena Image.png",0)

inp = cv2.resize(img, (200,200))

#%%

# Define Input field
u1 = 1*inp

# Define alpha matrix
alpha = np.sqrt(k**2 - 4*np.pi**2*(fx**2 + fy**2))

#first z
z = 1000e-8 # Try different distances and different sizes of rect
H = np.exp(1j*alpha*z)
U1 = fftshift(fft2(ifftshift(u1)))
U2 = U1 * H
u2 = fftshift(ifft2(ifftshift(U2)))

# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(abs(u2)**2)
plt.title("Fresnel diffraction pattern at z=" + str(z))
plt.colorbar()


#second z
z = 5000e-7 # Try different distances and different sizes of rect
H = np.exp(1j*alpha*z)
U1 = fftshift(fft2(ifftshift(u1)))
U2 = U1 * H
u2 = fftshift(ifft2(ifftshift(U2)))

# Plotting the result
plt.subplot(2, 2, 2)
plt.imshow(abs(u2)**2)
plt.title("Fresnel diffraction pattern at z=" + str(z))
plt.colorbar()


#third z
z = 1000e-6 # Try different distances and different sizes of rect
H = np.exp(1j*alpha*z)
U1 = fftshift(fft2(ifftshift(u1)))
U2 = U1 * H
u2 = fftshift(ifft2(ifftshift(U2)))

# Plotting the result
plt.subplot(2, 2, 3)
plt.imshow(abs(u2)**2)
plt.title("Fresnel diffraction pattern at z=" + str(z))
plt.colorbar()


#fourth z
z = 1000e-5 # Try different distances and different sizes of rect
H = np.exp(1j*alpha*z)
U1 = fftshift(fft2(ifftshift(u1)))
U2 = U1 * H
u2 = fftshift(ifft2(ifftshift(U2)))

# Plotting the result
plt.subplot(2, 2, 4)
plt.imshow(abs(u2)**2)
plt.title("Fresnel diffraction pattern at z=" + str(z))
plt.colorbar()


plt.show()






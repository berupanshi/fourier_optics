
"""

Fresnel and Fraunhofer diffraction patterns of 

various transmittance functions

"""

from IPython import get_ipython; 
get_ipython().run_line_magic('reset','-sf')


import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d




# Function to compute Fresnel Diffraction pattern

def ASM(inp_field, z):

    alpha = np.sqrt(k**2 - 4*np.pi**2*(fx**2 + fy**2))
    
    H = np.exp(1j*alpha*z)
    
    # Do the Fourier transform of input field
    
    U = fftshift(fft2(ifftshift(inp_field)))
    
    # Fourier transform of output field using ASM
    
    Uout = U * H
    
    # Finally, output field = Fresnel diffraction of input field
    
    u_out = fftshift(ifft2(ifftshift(Uout)))

    return u_out
  
  
# Function to compute Fraunhofer Diffraction pattern

def fourier(inp_field):

    
    # Do the Fourier transform of input field
    
    u_out = fftshift(fft2(ifftshift(inp_field)))


    return u_out
    
  




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

# Define transmittance functions/ aperture functions


######## Amplitude objects ##########


# Single Slit

wx = 4*dx 
wy = 40*dx 

t1 = np.where((abs(x) <= wx) * (abs(y) <= wy), 1.0, 0.0)


z = 100e-5


# Input field

u_inp = t1
u_fresn = ASM(u_inp, z)
u_fraun = fourier(u_inp)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow((u_inp), cmap='gray')
plt.colorbar()
plt.title('Input field')
plt.show()

plt.subplot(2, 2, 2)
plt.imshow(abs(u_fresn/np.max(u_fresn)), cmap='gray')
plt.colorbar()
plt.title('Fresnel-Diffraction pattern')

plt.subplot(2, 2, 3)
plt.imshow(abs(u_fraun/np.max(u_fraun)), cmap='gray')
plt.colorbar()
plt.title('Fraunhofer-Diffraction pattern')

# Plot 1D or line-profile by plotting intensity values
# of only x-axis i.e., N/2 th row and all columns of 2D Fraunhofer
# diffraction pattern

quan = abs(u_fraun/np.max(u_fraun))
plt.subplot(2, 2, 4)
plt.plot(range(N), quan[int(N/2),:])
plt.colorbar()
plt.title('1D analysis (Line profile of only x-axis)')
plt.grid()

plt.tight_layout()
#%%

# Double Slit


wx = 4*dx # half width i.e. rect spread over 20 samples
wy = 80*dx

b = 10 # Interslit Spacing in samples

double_delta = np.zeros([N,N], dtype='float')
double_delta[100,100 - int(np.round(b/2))] = 1
double_delta[100,100 + int(np.round(b/2))] = 1

t2 = convolve2d(t1, double_delta, mode='same')


z = 100e-5


# Input field

u_inp = t2
u_fresn = ASM(u_inp, z)
u_fraun = fourier(u_inp)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow((u_inp), cmap='gray')
plt.colorbar()
plt.title('Input field')
plt.show()

plt.subplot(2, 2, 2)
plt.imshow(abs(u_fresn/np.max(u_fresn)), cmap='gray')
plt.colorbar()
plt.title('Fresnel-Diffraction pattern')

plt.subplot(2, 2, 3)
plt.imshow(abs(u_fraun/np.max(u_fraun)), cmap='gray')
plt.colorbar()
plt.title('Fraunhofer-Diffraction pattern')


# Plot 1D or line-profile by plotting intensity values
# of only x-axis i.e., N/2 th row and all columns of 2D Fraunhofer
# diffraction pattern

quan = abs(u_fraun/np.max(u_fraun))
plt.subplot(2, 2, 4)
plt.plot(range(N), quan[int(N/2),:])
plt.colorbar()
plt.title('1D analysis (Line profile of only x-axis)')
plt.grid()


plt.tight_layout()
#%%


# Binary Amplitude Grating

freq = 1e5 # No. of lines/inch

t3 = np.sign(np.sin(freq * x))

z = 100e-5


# Input field

u_inp = t3
u_fresn = ASM(u_inp, z)
u_fraun = fourier(u_inp)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow((u_inp), cmap='gray')
plt.colorbar()
plt.title('Input field')
plt.show()

plt.subplot(2, 2, 2)
plt.imshow(abs(u_fresn/np.max(u_fresn)), cmap='gray')
plt.colorbar()
plt.title('Fresnel-Diffraction pattern')

plt.subplot(2, 2, 3)
plt.imshow(abs(u_fraun/np.max(u_fraun)), cmap='gray')
plt.colorbar()
plt.title('Fraunhofer-Diffraction pattern')


# Plot 1D or line-profile by plotting intensity values
# of only x-axis i.e., N/2 th row and all columns of 2D Fraunhofer
# diffraction pattern

quan = abs(u_fraun/np.max(u_fraun))
plt.subplot(2, 2, 4)
plt.plot(range(N), quan[int(N/2),:])
plt.colorbar()
plt.title('1D analysis (Line profile of only x-axis)')

plt.grid()

plt.tight_layout()
#%%


# Sinusoidal Amplitude Grating

amplitude = 1
freq = 1e5 # No. of lines/inch
phase_shift = 0
t4 = amplitude * np.sin(freq * x + phase_shift)

z = 100e-5


# Input field

u_inp = t4
u_fresn = ASM(u_inp, z)
u_fraun = fourier(u_inp)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow((u_inp), cmap='gray')
plt.colorbar()
plt.title('Input field')
plt.show()

plt.subplot(2, 2, 2)
plt.imshow(abs(u_fresn/np.max(u_fresn)), cmap='gray')
plt.colorbar()
plt.title('Fresnel-Diffraction pattern')

plt.subplot(2, 2, 3)
plt.imshow(abs(u_fraun/np.max(u_fraun)), cmap='gray')
plt.colorbar()
plt.title('Fraunhofer-Diffraction pattern')


# Plot 1D or line-profile by plotting intensity values
# of only x-axis i.e., N/2 th row and all columns of 2D Fraunhofer
# diffraction pattern

quan = abs(u_fraun/np.max(u_fraun))
plt.subplot(2, 2, 4)
plt.plot(range(N), quan[int(N/2),:])
plt.colorbar()
plt.title('1D analysis (Line profile of only x-axis)')


plt.grid()

plt.tight_layout()
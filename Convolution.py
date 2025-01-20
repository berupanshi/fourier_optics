"""

Convolution in 1D and 2D

"""



from IPython import get_ipython; 
get_ipython().run_line_magic('reset','-sf')


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


#%% Convolution of two rect functions in 1D

# Create the x-axis
x = np.arange(-5, 5, 0.02)

# Define two rectangular functions
rect1 = np.where(np.abs(x)<=1, 1, 0)
rect2 = np.where(np.abs(x)<=1, 1, 0)

# Perform the convolution using numpy.convolve
conv_result = np.convolve(rect1, rect2, mode='same')

# Try changing the parameters

# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, rect1)
plt.title("First Rectangular Function")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, rect2)
plt.title("Second Rectangular Function")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, conv_result)
plt.title("Convolution of rect1 and rect2")
plt.grid(True)

plt.show() 



#%% Convolution of two gaussian functions in 1D


a = 2
gaussian1 = np.exp(-(x**2)/(a**2))
b = 1
gaussian2 = np.exp(-(x**2)/(b**2))


# Perform the convolution using numpy.convolve
conv_result = np.convolve(gaussian1, gaussian2, mode='same')  # Normalize by time step

# Try changing the parameters

# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, gaussian1)
plt.title("First Gaussian Function")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, gaussian2)
plt.title("Second Gaussian Function")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, conv_result)
plt.title("Convolution of gaussian1 and gaussian2")
plt.grid(True)

plt.show()


#%% Convolution of two rect functions in 2D


p = 3 #  pixel size (in micrometre)

dx = p # lateral sample interval/pixel size
dy = p

N = 200 # N*N computational window

Lx = N*dx # Sampling period
Ly = N*dy 

m, n = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2)) 

x = m*dx # to take it to real dimension
y = n*dy

wx = 20*dx # half width i.e. rect spread over 20 samples
wy = 20*dy


rect1 = np.where((np.abs(x) <= wx) * (np.abs(y) <= wy), 1.0, 0.0)
rect2 = np.where((np.abs(x) <= wx) * (np.abs(y) <= wy), 1.0, 0.0)


# Perform the convolution using numpy.convolve
conv_result = convolve2d(rect1, rect2, mode='same')  # Normalize by time step

# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(rect1)
plt.title("First Rectangular Function")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(rect2)
plt.title("Second Rectangular Function")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(conv_result)
plt.title("Convolution of rect1 and rect2")
plt.colorbar()

plt.show()

#%% Surface plot for the above result

plt.figure(),
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x, y, abs(conv_result), cmap = plt.cm.cividis)
plt.show()


#%% Convolution of Gaussian and Double-Delta functions


double_delta = np.zeros_like(rect1, dtype='float')
double_delta[90,90] = 1; 
double_delta[90,110] = 1; 


a = 20   
gaussian2d = np.exp(-(x**2+y**2)/(a**2))


# Perform the convolution using convolve2d
conv_result = convolve2d(gaussian2d, double_delta, mode='same')


# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(gaussian2d)
plt.title("Guassian Function")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(double_delta)
plt.title("Double-Delta Function")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(conv_result)
plt.title("Convolution result")
plt.colorbar()

plt.show()



#%% Convolution of an image with gaussian function

img = plt.imread('Squares.png')


a = 20
gaussian2d = np.exp(-(x**2+y**2)/(a**2))


conv_result = convolve2d(img, gaussian2d, mode='same')


# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Loaded Image")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(gaussian2d, cmap='gray')
plt.title("Gaussian function")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(conv_result, cmap='gray')
plt.title("Convolution result")
plt.colorbar()

plt.show()


#%% Convolution theorem


####### LHS

#f1
rect1 = np.where((np.abs(x) <= wx) * (np.abs(y) <= wy), 1.0, 0.0)

#f2
a = 5
gaussian2d = np.exp(-(x**2+y**2)/(a**2))

conv_result = convolve2d(rect1, gaussian2d, mode='same')

lhs = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(conv_result)))

####### RHS

ft_rect2d = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rect1)))
ft_gaussian = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian2d)))

rhs = ft_rect2d*ft_gaussian

# Plotting the result
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(abs(lhs))
plt.title("LHS")

plt.subplot(1, 2, 2)
plt.imshow(abs(rhs))
plt.title("RHS")

plt.show()


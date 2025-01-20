from IPython import get_ipython; 
get_ipython().run_line_magic('reset','-sf')
import numpy as np
import matplotlib.pyplot as plt
#plt.close('all')

#%% 1D Function

x = np.arange(-3, 3, 0.01)
func1D = x**2
plt.figure(), plt.plot(x,func1D)

#%% Introduction to Meshgrid

x = np.arange(-3, 4)
y = np.arange(-7, 8)

X,Y = np.meshgrid(x,y)

#%% Example 2D Function
x = np.arange(-3, 3, 0.01)
y = np.arange(-7, 7, 0.01)

X,Y = np.meshgrid(x,y)

func2D = X**2 +Y**2

plt.figure(),
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, func2D, cmap = plt.cm.plasma)
plt.show()

#plt.plot(x, y, func)

#%% 2D
p = 3 #  pixel size (in micrometre)

dx = p # lateral sample interval/pixel size
dy = p

N = 200 # N*N computational window

Lx = N*dx #Sampling period
Ly = N*dy 
dfx = 1/Lx #Sampling interval in frequency domain
dfy = 1/Ly
m, n = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2)) 

x = m*dx #to take it to real dimension
y = n*dy

#%% 2D Rect function
wx = 1*dx #half width i.e. rect spread over 20 samples
wy = 20*dy
rect = (abs(x) <= wx)*(abs(y) <= wy) 

plt.figure(),plt.imshow(rect), plt.colorbar()

# Fourier-Transform
H1 = np.fft.fft2(rect)

H2 = np.fft.ifftshift(rect)

H3 = np.fft.fft2(np.fft.ifftshift(rect))

H4 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rect)))

H5 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H4)))

plt.figure(), 
plt.subplot(2,3,1),plt.imshow(rect), plt.colorbar(), plt.title("rect function")
plt.subplot(2,3,2), plt.imshow(abs(H2)), plt.colorbar(), plt.title("rect after ifftshift")
plt.subplot(2,3,3), plt.imshow(abs(H1)), plt.colorbar(), plt.title("ft_rect")
plt.subplot(2,3,4), plt.imshow(abs(H3)), plt.colorbar(), plt.title("ft of rect with ifftshift")
plt.subplot(2,3,5), plt.imshow(abs(H4)), plt.colorbar(), plt.title("fftshifted ft of rect with ifftshift")
plt.subplot(2,3,6), plt.imshow(abs(H5)), plt.colorbar(), plt.title("double fft")
#%% Rect Surface plot
plt.figure(),
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x, y, abs(H4), cmap = plt.cm.cividis)
plt.show()

#%% circle : pinhole
rad = 20
circ = np.sqrt(x**2+y**2)<rad
plt.figure(), 
plt.subplot(1,2,1),plt.imshow(circ), plt.colorbar(), plt.title("2D Circle function")
circ_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(circ)))
plt.subplot(1,2,2), plt.imshow(abs(circ_ft)), plt.colorbar(), plt.title("fourier transform of circ function")

#%% gaussian function
a = 15   
gaussian = np.exp(-(x**2+y**2)/(a**2))

plt.figure(), 
plt.subplot(1,2,1),plt.imshow(gaussian), plt.colorbar(), plt.title("2D Gaussian Function")
gaussian_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian)))
plt.subplot(1,2,2), plt.imshow(abs(gaussian_ft)), plt.colorbar(), plt.title("FT of 2D Gaussian Function")

#%% double delta
double_delta = np.zeros_like(rect, dtype='float')
double_delta[90,90] = 1; 
double_delta[90,110] = 1; 
plt.figure(), 
plt.subplot(1,2,1),plt.imshow(double_delta), plt.colorbar(), plt.title("Function")
double_delta_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(double_delta)))
plt.subplot(1,2,2), plt.imshow(abs(double_delta_ft)), plt.colorbar(), plt.title("FT of function")



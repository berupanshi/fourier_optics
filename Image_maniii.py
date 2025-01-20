# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:27:05 2024

@author: Rupanshi Pal
"""

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
img = plt.imread('square.jpg')
plt.imshow(img,cmap='plasma')
plt.show()
print(img)
print(type(img),'\n')
print(f"image data type:{img.dtype}\n")
print(f"image shape:{img.shape}\n")
print(f"Unique pixel values:{np.unique(img)}\n")

import options, util, warp, load_data
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

a = np.array([[1, 1, 1],
              [0, 0, 0],
              [0.3, 0.3, 0.3],
              [0.6, 0.6, 0.6]])
print(a.shape)
print(a[2,1])
plt.imshow(a, cmap='gray')
plt.show()
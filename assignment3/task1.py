import numpy as np
from scipy.ndimage import convolve

A = np.array([[2, 1, 2, 3, 1],
              [3, 9, 1, 1, 4],
              [4, 5, 0, 7, 0]])

K = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])


O = convolve(A,K,mode='constant')
print(f"{O}")
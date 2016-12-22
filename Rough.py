from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import nltk
import collections
import math
import os
import random
import zipfile
import numpy as np, tensorflow as tf

batch_size=128

batch = np.ones(shape=(batch_size), dtype=np.int32)
labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
a= 36 % 3668

matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
ids = np.array([0, 5, 17, 33])
print(matrix[ids])  # prints a matrix of shape [4, 64]

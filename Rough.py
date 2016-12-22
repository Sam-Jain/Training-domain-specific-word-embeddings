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
import matplotlib.pyplot as plt

batch_size=128

a= 36 % 3668
average_loss = 0
m = [average_loss]
for i in range(0,1000,1):
    average_loss += -2
    m.append(average_loss)

print(m)
plt.plot(m)
plt.show()
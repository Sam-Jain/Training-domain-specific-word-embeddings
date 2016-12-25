from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

#from multiprocessing import Pool, Value, Array

import nltk
import collections
import math
import os
import random
import zipfile
import matplotlib.pyplot as plt
from multiprocessing import Array


def domain_corpus(di):
    f = open(di).read().lower().split()
    domain_word_list = []
    for word in f:
        domain_word_list.append(word)
    return(domain_word_list)


domain_vocab = domain_corpus('domain_words.txt')
print(domain_vocab)

MyOper = set(['AND', 'OR', 'NOT'])
MyList = set(['c1', 'c2', 'NOT', 'c3'])

while not MyList.isdisjoint(MyOper):
    print("No boolean Operator")


"""You can't generally say for x not in y because there are an infinite number of objects that aren't in y.

For your specific case, I think something like this might work:

[x for x in range(0, n) if x not in y]"""
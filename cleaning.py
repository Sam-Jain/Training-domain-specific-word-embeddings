import argparse
import math
import struct
import sys
import time
import warnings
import pickle
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
from multiprocessing import Array
from bs4 import BeautifulSoup



fi = 'test_data.txt'

printable = set(string.printable)
# stop words from nltk
stop_words = set(stopwords.words("english"))

file_con = open(fi).read().lower()
file_content = filter(lambda x: x in printable, file_con)

example_words = word_tokenize(file_content)
# removing punctuations
example_words = filter(lambda x: x not in string.punctuation, example_words)
# removing stop_words
cleaned_text = filter(lambda x: x not in stop_words, example_words)
print(cleaned_text)
cleaned_t = " ".join(cleaned_text)
f = open('cleaned_test_data.txt', 'w')
f.write(cleaned_t)

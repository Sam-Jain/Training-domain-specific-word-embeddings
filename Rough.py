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

'''
def read_data(filename):
  with open(filename,'r') as f:
    data = f.read().split()
  print(data)

for i in range(3):
  datum = (str(i)+'_10.txt')
  read_data(datum)
'''

filename = 'test_data.txt'

# Read the data into a list of strings.
def read_data(filename):
  with open(filename,'r') as f:
      data = f.read().split()
  return data

domain_corpus = 'domain_words.txt'
domain_words = read_data(domain_corpus)
print('this is Domain data',domain_words)

words = read_data(filename)
print('Data size', len(words))
vocabulary_size = len(words)

print('this is words:',words)
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict() #the above line gives us the ('word',no_of_occurences) in the list count
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('data',data)
print('count',count)
print('dictionary',dictionary)
print('reverse_dictionary',reverse_dictionary)

data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
#returns batch and labels
def generate_batch(batch_size, num_skips, window_size):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * window_size
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  #print(batch)
  #print(labels)
  span = 2 * window_size + 1  # [ window_size target window_size ]
  buffer = collections.deque(maxlen=span)
  #print(span)
  #print(buffer)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data) #?????len(data) is always
    #print(buffer)
    #print(data_index)
    #print(len(data))
  for i in range(batch_size // num_skips):
    target = window_size  # target label at the center of the buffer
    targets_to_avoid = [window_size]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
        #print('this is the target',target)
      targets_to_avoid.append(target)
      #print('target to avoid',targets_to_avoid,j)
      batch[i * num_skips + j] = buffer[window_size]
      labels[i * num_skips + j, 0] = buffer[target]
      #print('this is the batch', batch)
      #print('this is the label', labels)
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data) #same as above????????
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, window_size=1)
#************************************LINE INCLuded*****************************************************************************


batch, labels_from_domain = generate_batch(batch_size=1, num_skips=1, window_size=1)


print('these are the labels from the domain', labels_from_domain)
#print(batch)
#print(labels)


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
window_size = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
#NOTE:np.random.choice(a,size='output_shape',replace='with or without replacement') generates a random sample from a which is a 1d array



graph = tf.Graph()
with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  #*******************************************************************************************************************************
  train_labels_domain = tf.placeholder(tf.int32, shape=[batch_size,1])
  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) #both are initialized above
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  #loss = tf.reduce_mean
  '''  'x' is [[1., 1.]
               [2., 2.]]
       tf.reduce_mean(x) ==> 1.5
       tf.reduce_mean(x, 0) ==> [1.5, 1.5]
       tf.reduce_mean(x, 1) ==> [1.,  2.]'''

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels_domain,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer
  init = tf.global_variables_initializer()







# Step 5: Begin training.
training_epochs = 50001

with tf.Session(graph=graph) as session:
  #writer = tf.train.FileWriter("logs/", session.graph)
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  average_plot = []
  for step in xrange(training_epochs):
    batch_inputs, labels_from_domain = generate_batch(
        batch_size, num_skips, window_size)
    feed_dict = {train_inputs: batch_inputs, train_labels_domain: labels_from_domain }

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    average_plot.append(average_loss)

    if step % 100 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    if step % 10000 == 0:
        sim = similarity.eval()
        '''for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)'''
    final_embeddings = normalized_embeddings.eval()
    #print(final_embeddings)


plt.plot(average_plot)
plt.show()

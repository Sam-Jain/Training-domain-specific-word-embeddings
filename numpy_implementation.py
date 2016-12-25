import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

#used in sort and Vocab functions dont need to dive deep
class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0 #count of word in the list
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    # It returns the vocab_items, vocab_hash and word_count
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fii = open(fi).read().lower()
        fi = open(fi,'r')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items  # List of VocabItem objects
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print 'Total words in training file: %d' % self.word_count
        print 'Total bytes in training file: %d' % self.bytes
        print 'Vocab size: %d' % len(self)
        print 'this is vocab items', self.vocab_items
        print 'this is vocab hash', self.vocab_hash

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash
    #Takes the min_count as input which are the words which will be removed with less than min_count
    # occurences and returns the vocab_hash dictionary
    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print 'this is the other vocab hash', self.vocab_hash
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]#******************************************


#it gives back a table which is a list of indices from which negative samples are drawn
class UnigramTable:
    """
    A list of indices of tokens (List of indices is stored in
     self.path of VocabItem) in the vocab following a power law distribution,
    used to draw negative samples.

    """
    pie = 0.5
    #unigram table for (w,c) both belonging to domain_vocab
    def case1(self, vocab):
        xs = np.random.uniform(low=0, high=1)
        if 0.5 > xs:
            smoothing_parameter = 0.75
            norm1 = sum([math.pow(c.count, smoothing_parameter)for c in vocab])

            table_size1 = 100000000
            table1 = np.zeros(table_size1, dtype=np.int32)

            print 'Filling the unigram table for case1'
            p1 = 0 #Cumulative probability for case1
            i1 = 0
            for j1, unigram1 in enumerate(vocab):
                p1 += float(math.pow(unigram1.count, smoothing_parameter)) / norm1
                while i1 < table_size1 and float(i1) / table_size1 <p1:
                    table[i1] = j1
                    i1 += 1

            self.table1
        else:
            smoothing_parameter = 0.75
            norm1 = sum([math.pow(c.count, smoothing_parameter) for c in vocab])

            table_size1 = 1e8
            table1 = np.zeros(table_size1, dtype=np.int32)

            print 'Filling the unigram table for case1'
            p1 = 0  # Cumulative probability for case1
            i1 = 0
            for j1, unigram1 in enumerate(vocab):
                p1 += float(math.pow(unigram1.count, smoothing_parameter)) / norm1
                while i1 < table_size1 and float(i1) / table_size1 < p1:
                    table[i1] = j1
                    i1 += 1

            self.table1
    #this will be the def case2**
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) #Normalizing constant***************************************************************

        table_size = 1e8  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.int32)

        print 'Filling unigram table'
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def case3(self, vocab):
        """Code has to written over here"""
    #This is used as the sample to pick the neagtive samples
    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices] #**************************************************************************************************




def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

#returns syn0 and syn1
#Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
# And syn1 with zeros bith as ctype array
def init_net(dim, vocab_size):

    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)
    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

#This function trains the vocab
def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    fi.seek(start)
    print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)
    alpha = starting_alpha
    word_count = 0
    last_word_count = 0
    #tell returns the current position of the read/write pointer within the file.
    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line from vocab_hash
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])
        sentence = vocab.indices(line.split())
        for sent_po, tokenz in enumerate(sentence):
            if tokenz in vocab:
                """code will be filled here"""
                print 'hello'
            elif tokenz not in vocab:
                for sent_pos, token in enumerate(sent):
                    if word_count % 10000 == 0:
                        global_word_count.value += (word_count - last_word_count)
                        last_word_count = word_count

                        # Recalculate alpha
                        alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                        if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                        # Print progress info
                        sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                         (alpha, global_word_count.value, vocab.word_count,
                                          float(global_word_count.value) / vocab.word_count * 100))
                        sys.stdout.flush()

                        # getting the context
                        # Randomize window size, where win is the max window size
                        current_win = np.random.randint(low=1, high=win + 1)
                        # sent_pos is the index of a particular word of the sentence which is sent
                        context_start = max(sent_pos - current_win, 0)
                        context_end = min(sent_pos + current_win + 1, len(sent))
                        # ek dum correct
                        context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?

                    for context_word in context:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)
                        # Compute neu1e and update syn1
                        if neg > 0: # neg number of negative samples are being called from sample function line:141
                            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                        #else:
                        #    classifiers = zip(vocab[token].path, vocab[token].code)
                        for target, label in classifiers: #How does this work?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                            #this is where context_word is dot multiplied with target
                            z = np.dot(syn0[context_word], syn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * syn1[target]  # Error to backpropagate to syn0
                            syn1[target] += g * syn0[context_word]  # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e

                word_count += 1

        else:
            """Code to be filled in here"""

    # Print progress info
    global_word_count.value += (word_count - last_word_count) #last_word_count = word_count line194
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()

#for saving the output
def save(vocab, syn0, fo, binary):
    print 'Saving model to', fo
    dim = len(syn0[0])
    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), dim))
    for token, vector in zip(vocab, syn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

#This is the initializer for the pool function i.e. called when pool is called line 276
def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    print 'this is args', args
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        #Create a numpy array from a ctypes array or a ctypes POINTER. The numpy array shares the memory with the ctypes object.
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

#for training
def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)
    print('this is vocab', vocab)
    domain_vocab = Vocab(di, min_count)
    # Init net
    syn0, syn1 = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        table = UnigramTable(vocab)
    # Begin training using num_processes workers
    t0 = time.time()
    '''******pool is just used to start parallel processes the arguments are as given******'''
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, fi))
    #this maps the train_process function with the number of processes
    print pool.map(train_process, range(num_processes))
    t1 = time.time()

    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file the function is defined above
    save(vocab, syn0, fo, binary)


di = 'domain_words.txt'
fi= 'test_data.txt'
fo= 'final.txt'
train(fi, fo, 0, 100, 300, 0.01, 2, 5, 1, 0)

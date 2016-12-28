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
        self.count = 0     #count of word in the list
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
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

#returns a list of words form domain_corpus
def domain_corpus(di):
    f = open(di).read().lower().split()
    domain_word_list = []
    for word in f:
        domain_word_list.append(word)
    return(domain_word_list)

#it gives back a table which is a list of indices from which negative samples are drawn
class UnigramTable1:

    def __init__(self, vocab):
        #unigram table for (w,c) both belonging to domain_vocab
        pie = 0.5
        smoothing_parameter = 0.75
        xs = np.random.uniform(low=0, high=1)
        table_size = 1e2
        domain_vocab = domain_corpus(di)
        if xs < pie:
            norm = sum([math.pow(c.count, smoothing_parameter) for c in range(len(domain_vocab)) if c not in domain_vocab])
            #norm = sum([math.pow(c.count, smoothing_parameter) for c in domain_vocab])
            table = np.zeros(table_size, dtype=np.uint32)

            print 'Filling the unigram table for case1'
            p = 0 #Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = j
                    i += 1

            self.table = table
        else:
            norm = sum([math.pow(t.count, smoothing_parameter) for t in domain_vocab])
            table = np.zeros(table_size, dtype=np.uint32)

            print 'Filling the unigram table for case1'
            p = 0  # Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = j
                    i += 1
            self.table = table

            # This is used as the sample to pick the neagtive samples

    def sample1(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

#unigram table for the second case
class UnigramTable2:

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        table_size = 1e2  # Length of the unigram table

        table = np.zeros(table_size, dtype=np.uint32)
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        print 'Filling unigram table for case2'
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
    def sample2(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

#unigram table for the third case
class UnigramTable3:
    def __init__(self, vocab):
        """This is the third case"""
        pie = 0.5
        smoothing_parameter = 0.75
        z = np.random.uniform(low=0, high=1)
        table_size = 1e2
        domain_vocab = domain_corpus(di)
        if z < pie:
            #norm = sum([math.pow(c.count, smoothing_parameter) for c in range(len(domain_vocab)) if c not in domain_vocab])
            norm = sum([math.pow(c.count, smoothing_parameter) for c in domain_vocab])
            table = np.zeros(table_size, dtype=np.uint32)

            print 'Filling the unigram table for case3'
            p = 0 #Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size <p:
                    table[i] = j
                    i += 1

            self.table = table
        else:
            norm = sum([math.pow(c.count, smoothing_parameter) for c in domain_vocab])
            table = np.zeros(table_size, dtype=np.uint32)

            print 'Filling the unigram table for case3'
            p = 0  # Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = j
                    i += 1
            self.table = table

    def sample3(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

#returns w_ip_hidden and w_op_hidden
#Init w_ip_hidden with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
# And w_op_hidden with zeros bith as ctype array
def init_net(dim, vocab_size):

    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    w_ip_hidden = np.ctypeslib.as_ctypes(tmp)
    w_ip_hidden = Array(w_ip_hidden._type_, w_ip_hidden, lock=False)
    # Init w_op_hidden with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    w_op_hidden = np.ctypeslib.as_ctypes(tmp)
    w_op_hidden = Array(w_op_hidden._type_, w_op_hidden, lock=False)

    return (w_ip_hidden, w_op_hidden)

#This function trains the vocab
def train_process():
    # Set fi to point to the right chunk of training file
    start = 0
    end = vocab.bytes
    fi.seek(start)
    print 'Worker beginning training at %d, ending at %d' % (start, end)
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
        domain_vocab = domain_corpus(di)
        for sent_pos, token in enumerate(sent):
            #this will store the current word
            current_token = token
            #this controls how many time should the updates be made ASK CONFIRM
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count
                # overall value of alpha is decreasing
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

            sys.stdout.flush()
            # getting the context
            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win + 1)
            # sent_pos is the index of a particular word of the sentence which is sent
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?
            for context_word in context:
                #This will hold the current context for the current_token
                current_context = context
                if current_token and current_context in domain_vocab:
                    print('this is current token and context', current_token, current_context)
                    if word_count % 10000 == 0:
                        '''global_word_count.value += (word_count - last_word_count)
                        last_word_count = word_count'''
                        # overall value of alpha is decreasing
                        alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                        if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001
                    """This is the first case where (w,c) both belong in Nu"""
                    table = UnigramTable1(vocab)
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)
                    # Compute neu1e and update w_op_hidden
                    if neg > 0:  # neg number of negative samples are being called from sample function line:141
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample1(neg)]
                    for target, label in classifiers:  # It iterates over tuple level ie first element of the tuple will be
                        # the target and second will be label!
                        # this is where context_word is dot multiplied with target
                        z = np.dot(w_ip_hidden[context_word], w_op_hidden[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * w_op_hidden[target]  # Error to backpropagate to w_ip_hidden
                        w_op_hidden[target] += g * w_ip_hidden[context_word]  # Update w_op_hidden
                        # Update w_ip_hidden
                        w_ip_hidden[context_word] += neu1e

                elif current_token and current_context not in domain_vocab:
                    """This is the case where none of (w,c) belongs to Nu"""
                    if word_count % 10000 == 0:
                        global_word_count.value += (word_count - last_word_count)
                        last_word_count = word_count
                        # overall value of alpha is decreasing
                        '''alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                        if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001'''
                    table = UnigramTable2(vocab)
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)
                    # Compute neu1e and update w_op_hidden
                    if neg > 0:  # neg number of negative samples are being called from sample function line:141
                       classifiers = [(token, 1)] + [(target, 0) for target in table.sample2(neg)]
                    for target, label in classifiers: #It iterates over tuple level ie first element of the tuple will be
                                                      #the target and second will be label!
                        #this is where context_word is dot multiplied with target
                        z = np.dot(w_ip_hidden[context_word], w_op_hidden[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * w_op_hidden[target]  # Error to backpropagate to w_ip_hidden
                        w_op_hidden[target] += g * w_ip_hidden[context_word]  # Update w_op_hidden
                        # Update w_ip_hidden
                        w_ip_hidden[context_word] += neu1e
                '''
                else:
                    """ This is the third case where either one of them (w,c) does not belong to Nu """
                    # Init neu1e with zeros
                    table = UnigramTable3(vocab)
                    neu1e = np.zeros(dim)
                    # Compute neu1e and update w_op_hidden
                    if neg > 0:  # neg number of negative samples are being called from sample function line:141
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample3(neg)]
                    for target, label in classifiers:  # It iterates over tuple level ie first element of the tuple will be
                        # the target and second will be label!
                        # this is where context_word is dot multiplied with target
                        z = np.dot(w_ip_hidden[context_word], w_op_hidden[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * w_op_hidden[target]  # Error to backpropagate to w_ip_hidden
                        w_op_hidden[target] += g * w_ip_hidden[context_word]  # Update w_op_hidden
                        # Update w_ip_hidden
                        w_ip_hidden[context_word] += neu1e'''
        word_count += 1
    # Print progress info
    global_word_count.value += (word_count - last_word_count)  # last_word_count = word_count line194
    sys.stdout.write("\rAlpha: %f Progress:" % alpha)
    sys.stdout.flush()
    fi.close()

#for saving the output
def save(vocab, w_ip_hidden, fo, binary):
    print 'Saving model to', fo
    dim = len(w_ip_hidden[0])
    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(w_ip_hidden), dim))
    for token, vector in zip(vocab, w_ip_hidden):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

#This is the initializer for the pool function i.e. called when pool is called line 276
def global_func(*args):
    global vocab, w_ip_hidden, w_op_hidden, table, cbow, neg, dim, starting_alpha
    global win, global_word_count, fi #removed num_processes

    vocab, w_ip_hidden_tmp, w_op_hidden_tmp, table, cbow, neg, dim, starting_alpha, win, global_word_count = args[:-1]
    print 'this is args', args

    fi = open(args[-1], 'r')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        #Create a numpy array from a ctypes array or a ctypes POINTER. The numpy array shares the memory with the ctypes object.
        w_ip_hidden = np.ctypeslib.as_array(w_ip_hidden_tmp)
        w_op_hidden = np.ctypeslib.as_array(w_op_hidden_tmp)

#for training
def train(fi, fo, cbow, neg, dim, alpha, win, min_count, binary):
    # Read train file to init vocab
    domain_vocab = domain_corpus(di)
    vocab = Vocab(fi, min_count)
    print('this is vocab', vocab)
    # Init net
    w_ip_hidden, w_op_hidden = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        #table = UnigramTable(2, vocab)
    # Begin training using num_processes workers
    t0 = time.time()
    '''******pool is just used to start parallel processes the arguments are as given******'''

    global_func(vocab, w_ip_hidden, w_op_hidden, table, cbow, neg, dim, alpha, win, global_word_count, fi)

    #pool = Pool(processes=num_processes, initializer=__init_process,
    #            initargs=(vocab, w_ip_hidden, w_op_hidden, table, cbow, neg, dim, alpha,
    #                      win, num_processes, global_word_count, fi))
    #this maps the train_process function with the number of processes
    print train_process()
    #print pool.map(train_process, range(num_processes))
    t1 = time.time()

    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file the function is defined above
    save(vocab, w_ip_hidden, fo, binary)


di = 'domain_words.txt'
fi = 'test_data.txt'
fo = 'final.txt'
train(fi, fo, 0, 10, 300, 0.5, 2, 1, 0)

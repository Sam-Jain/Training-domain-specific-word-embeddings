import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Array


class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fii = open(fi).read().lower()
        fi = open(fi, 'r')
        #fi = open(fi, 'r')

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

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

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

        print
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

def domain_corpus(di):
    f = open(di).read().lower().split()
    domain_word_list = set()
    for word in f:
        domain_word_list.add(word)
    return(domain_word_list)

class UnigramTable1:

    def __init__(self, vocab):
        #unigram table for (w,c) both belonging to domain_vocab
        pie = 0.5
        smoothing_parameter = 0.75
        xs = np.random.uniform(low=0, high=1)
        table_size = 1e8
        domain_vocab = domain_corpus(di)
        table1 = np.zeros(table_size, dtype=np.uint32)
        if xs < pie:
            norm = len(domain_vocab)
            #norm = float(sum([math.pow(c.count, smoothing_parameter)for c in vocab]))

            print 'Filling the unigram table for case1'
            p = 0 #Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size < p:
                    table1[i] = j
                    i += 1

            self.table1 = table1
        else:
            norm = float(sum([math.pow(t.count, smoothing_parameter) for t in domain_vocab]))
            print 'Filling the unigram table for case1'
            p = 0  # Cumulative probability for case1
            i = 0
            for j, unigram in enumerate(vocab):
                p += float(math.pow(unigram.count, smoothing_parameter)) / norm
                while i < table_size and float(i) / table_size < p:
                    table1[i] = j
                    i += 1
            self.table1 = table1

            # This is used as the sample to pick the neagtive samples

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table1), size=count)
        return [self.table1[i] for i in indices]

class UnigramTable2:
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        table_size = 1e8  # Length of the unigram table
        table2 = np.zeros(table_size, dtype=np.uint32)

        print 'Filling the unigram table for case2'
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table2[i] = j
                i += 1
        self.table2 = table2

    def sample(self, count):
        iters = np.random.randint(low=0, high=len(self.table2), size=count)
        return [self.table2[i] for i in iters]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)


def train_process():

    start = 0
    end = vocab.bytes
    fi.seek(start)
    global_word_count = 0
    alpha = starting_alpha
    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue
        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            current_token = token
            if word_count % 10000 == 0:
                global_word_count += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count) / vocab.word_count)
                if alpha < starting_alpha * 0.0001:
                    alpha = starting_alpha * 0.0001

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]

            for context_word in context:
                current_context = context_word
                if current_token and current_context in domain_vocab:
                    print 'hello'
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)
                    # Compute neu1e and update syn1
                    classifiers = [(token, 1)] + [(target, 0) for target in table1.sample(neg)]
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = np.log(sigmoid(z)) * neg
                        g = alpha * (label - p)

                        neu1e += g * syn1[target]  # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word]  # Update syn1
                    # Update syn0
                    syn0[context_word] += neu1e
                elif current_token and current_context not in domain_vocab:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)
                    # Compute neu1e and update syn1
                    classifiers = [(token, 1)] + [(target, 0) for target in table2.sample(neg)]
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)

                        neu1e += g * syn1[target]  # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word]  # Update syn1
                    # Update syn0
                    syn0[context_word] += neu1e
                else:
                    z = np.random.uniform(low=0, high=1)
                    pie0 = 0.5
                    if z < pie0:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)
                        # Compute neu1e and update syn1
                        classifiers = [(token, 1)] + [(target, 0) for target in context]

                        for target, label in classifiers:
                            z = np.dot(syn0[context_word], syn1[target])
                            p = np.log(sigmoid(-z))
                            g = alpha * (label - p)

                            neu1e += g * syn1[target]  # Error to backpropagate to syn0
                            syn1[target] += g * syn0[context_word]  # Update syn1
                        # Update syn0
                        syn0[context_word] += neu1e
                    else:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)
                        # Compute neu1e and update syn1
                        classifiers = [(token, 1)] + [(target, 0) for target in context]

                        for target, label in classifiers:
                            z = np.dot(syn0[context_word], syn1[target])
                            p = np.log(sigmoid(z))
                            g = alpha * (label - p)

                            neu1e += g * syn1[target]  # Error to backpropagate to syn0
                            syn1[target] += g * syn0[context_word]  # Update syn1
                        # Update syn0
                        syn0[context_word] += neu1e


            word_count += 1

    # Print progress info
    global_word_count += (word_count - last_word_count)
    # Print progress info
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count, vocab.word_count,
                      float(global_word_count) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()


def save(vocab, syn0, fo):
    print 'Saving model to', fo
    dim = len(syn0[0])

    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), dim))
    for token, vector in zip(vocab, syn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()


def global_func(*args):
    global vocab, domain_vocab, syn0, syn1, table1, table2, table3, neg, dim, starting_alpha
    global win, global_word_count, fi

    vocab, domain_vocab, syn0_tmp, syn1_tmp, table1, table2, table3, neg, dim, starting_alpha, win, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

    warnings.filterwarnings('ignore')


def train(fi, fo, neg, dim, alpha, win, min_count):
    """#STEP1"""
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)
    domain_vocab = domain_corpus(di)
    """#STEP2"""
    # Init net
    syn0, syn1 = init_net(dim, len(vocab))
    """#STEP3"""
    global_word_count = 0
    """#STEP4"""
    table1 = None
    table2 = None
    table3 = None
    print 'Initializing unigram table'
    table1 = UnigramTable1(vocab)
    table2 = UnigramTable2(vocab)
    global_func(vocab, domain_vocab, syn0, syn1, table1, table2, table3, neg, dim, alpha, win, global_word_count, fi)
    """"#STEP5"""
    # Begin training using num_processes workers
    t0 = time.time()
    train_process()
    t1 = time.time()
    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'
    """#STEP6"""
    # Save model to file
    save(vocab, syn0, fo)
    print syn0


fi = 'test_data.txt'
fo = 'test6.txt'
di = 'domain_words.txt'
train(fi, fo, 100, 300, 0.01, 1, 1)

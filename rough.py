from bs4 import BeautifulSoup
import string
import codecs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#soup = BeautifulSoup(open('20040102'), "lxml")
#corpus = soup.get_text()
#corpus_c = corpus.decode('utf-8')
out = 'out.txt'

corpus = open('test_data.txt', )
stop_words = set(stopwords.words("english"))

example_words = word_tokenize(corpus)
#removing punctuations
example_words = (filter(lambda x: x notstring.punctuation, example_words))
#removing stop_words
cleaned_text = (filter(lambda x: x notstop_words, example_words)).lower()


f = open(out, 'w')
f.write(cleaned_text)

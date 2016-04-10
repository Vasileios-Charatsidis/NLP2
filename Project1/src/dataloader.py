import codecs
import re
from nltk.stem import PorterStemmer

def __preprocess(sentence, stemmer):
  # Remove whitespace at the end of the sentence (line)
  sentence = re.sub('[ \t\n]*$','',sentence)
  # Convert to lower and split by whitespace
  words = sentence.lower().split()
  # Stem words (not sure if we should do that)
  #words = map(lambda word: stemmer.stem(word), words)
  return words

#1 underscore means you are not enouraged to use this outside of this module
def _read(file_name):
  f = codecs.open(file_name, mode='r', encoding='utf-8')
  vocab = set()
  corpus = list()
  stemmer = PorterStemmer()
  for sentence in f:
    # TODO: Get word roots/lemmas. Should be better
    sentence_words = __preprocess(sentence, stemmer)
    corpus.append(sentence_words)
    map(lambda word: vocab.add(word), sentence_words)
  return corpus, vocab

#2 underscores mean you are really not supposed to use this outside of this module
def __print_corpus(corpus):
  for sentence in corpus:
    print sentence

def read_data(english_file_name, french_file_name):
  english, e_vocab = _read(english_file_name)
  french, f_vocab = _read(french_file_name)
  
#  __print_corpus(english)
#  __print_corpus(french)
  
  return english, french, e_vocab, f_vocab
  

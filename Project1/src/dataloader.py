import codecs
import re
from nltk.stem import PorterStemmer

def _preprocess(sentence, stemmer):
  # Remove whitespace at the end of the sentence (line)
  sentence = re.sub('[ \t\n]*$','',sentence)
  # Convert to lower and split by whitespace
  words = sentence.lower().split()
  # Stem words (not sure if we should do that)
  #words = map(lambda word: stemmer.stem(word), words)
  return words

def _read(file_name):
  f = codecs.open(file_name, mode='r', encoding='utf-8')
  vocab = set()
  corpus = list()
  stemmer = PorterStemmer()
  for sentence in f:
    # TODO: Get word roots/lemmas. Should be better
    sentence_words = _preprocess(sentence, stemmer)
    corpus.append(sentence_words)
    map(lambda word: vocab.add(word), sentence_words)
  return corpus, vocab

def _print_corpus(corpus):
  for sentence in corpus:
    print sentence

def read_data(english_file_name, french_file_name):
  english, e_vocab = _read(english_file_name)
  french, f_vocab = _read(french_file_name)
  
#  _print_corpus(english)
#  _print_corpus(french)
  
  return english, french, e_vocab, f_vocab
  

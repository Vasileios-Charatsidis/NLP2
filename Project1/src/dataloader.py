import codecs
import re
from collections import defaultdict
from nltk.stem import PorterStemmer

def _preprocess(sentence, stemmer):
  # Remove whitespace at the end of the sentence (line)
  sentence = re.sub('[ \t\n]*$','',sentence)
  # Convert to lower and split by whitespace
  words = sentence.lower().split()
  # Stem words (not sure if we should do that)
  #words = map(lambda word: stemmer.stem(word), words)
  return words

def read(file_name):
  f = codecs.open(file_name, mode='r', encoding='utf-8')
  vocab = set()
  corpus = list()
  stemmer = PorterStemmer()
  for sentence in f:
    # TODO: Get word roots/lemmas. Should be better
    sentence_words = _preprocess(sentence, stemmer)
    corpus.append(sentence_words)
    map(lambda word: vocab.add(word), sentence_words)
  f.close()
  return corpus, vocab

def _print_corpus(corpus):
  for sentence in corpus:
    print sentence

def read_train_data(english_file_name, french_file_name):
  english, e_vocab = read(english_file_name)
  french, f_vocab = read(french_file_name)
  
#  _print_corpus(english)
#  _print_corpus(french)
  
  return english, french, e_vocab, f_vocab

def read_sentence_alignments(alignment_file_name):
  alignment_file = open(alignment_file_name, 'r')
  sure_alignments = defaultdict(set)
  prob_alignments = defaultdict(set)
  for alignment in alignment_file:
    #0001 1 1 S
    tokens = alignment.strip('\n').split()
    sentence_no = int(tokens[0])
    e = int(tokens[1])
    f = int(tokens[2])
    if len(tokens) <= 3 or 's' == tokens[3].lower():
      sure_alignments[sentence_no].add( (e, f) )
    else:
      prob_alignments[sentence_no].add( (e, f) )
  alignment_file.close()
  return sure_alignments, prob_alignments

def _read_alignments(alignment_file_name):
  alignment_file = open(alignment_file_name, 'r')
  sure_alignments = set()
  prob_alignments = set()
  for alignment in alignment_file:
    #0001 1 1 S
    tokens = alignment.strip('\n').split()
    sentence_no = int(tokens[0])
    e = int(tokens[1])
    f = int(tokens[2])
    if 's' == tokens[3].lower():
      sure_alignments.add( (sentence_no, e, f) )
    else:
      prob_alignments.add( (sentence_no, e, f) )
  alignment_file.close()
  return sure_alignments, prob_alignments

def read_test_data(english_file_name, french_file_name, alignment_file_name):
  english, _ = read(english_file_name)
  french, _ = read(french_file_name)
  sure, probable = _read_alignments(alignment_file_name)
  return english, french, sure, probable

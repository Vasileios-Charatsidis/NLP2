import numpy as np
import math
import gc

import time

class IBM:

  #Define a sensible null word
  null_word = 0

  def __init__(self, e_vocab, f_vocab):
    start = time.time()
    self.e_vocab = self.__init_vocab(e_vocab,True)
    print 'initialized e_vocab', time.time() - start
    start = time.time()
    self.f_vocab = self.__init_vocab(f_vocab,False)
    print 'initialized f_vocab', time.time() - start

  def __init_vocab(self, set_vocab, add_null_word):
    vocab = dict()
    i = 1 if add_null_word else 0
    for word in set_vocab:
      vocab[word] = i
      i += 1
    return vocab

  def __compute_AER(self, alignments, sure, possible):
    assert(len(alignments) == len(sure))
    assert(len(alignments) == len(possible))
    aer = 0
    for sentence_no in range(len(alignments)):
      s_alignments  = alignments[sentence_no]
      s_sure        = sure[sentence_no]
      s_possible    = possible[sentence_no]
      # just in the case of sentences with no alignments
      if 0 == len(s_alignments) + len(s_sure):
        continue
      correct_sure = len(s_alignments.intersection(s_sure))
      correct_possible = len(s_alignments.intersection(s_possible))
      aer += 1 - float(correct_sure + correct_possible) / float(len(s_alignments) + len(s_sure))
    return aer

  def __split_data(self, english, french):
    sentence_count = len(english)
    sentence_nos = np.arange(sentence_count)
    #shuffles in-place
    np.random.shuffle(sentence_nos)
    #take 80% of data as train data, the rest is for validation
    training_count = math.floor(0.8 * sentence_count)
    training = sentence_nos[:training_count]
    validation = sentence_nos[training_count:]
    return training, validation

  # List of sentences
  # sentence - list of words
  def train(self, english, french, iterations):
    assert(len(english) == len(french))
    # Initialize thetas
    start = time.time()
    self.thetas = self.__random_initialize_thetas(english, french)
    print 'initialized thetas', time.time() - start

    # save best thetas after every iteration
    best_thetas = self.thetas
    best_log_likelihood = float('-inf')

    # Run EM
    for i in range(iterations):
      i_start = time.time()

      #possibly split the data into train and validation data for cross-validation?
      #training, validation = self.__split_data(english, french) #training, validation - arrays of indices of sentences
      start = time.time()
      self.__iteration(english, french)
      print 'iteration finished', time.time() - start
      
      start = time.time()
      log_likelihood, alignments = self.__compute_log_like_and_alignments(english, french)
      print 'computed log-likelihood', time.time() - start

      start = time.time()
      # testing with the same alignments until we get anotated data
      aer = self.__compute_AER(alignments, alignments, alignments)
      print 'computed AER', time.time() - start
      
      start = time.time()
      if best_log_likelihood < log_likelihood:
        best_thetas = self.thetas
        best_log_likelihood = log_likelihood

      print 'Iteration', i, 'Time:',time.time() - i_start,'s','Log-likelihood',log_likelihood,'AER',aer
    
    #recover best thetas
    self.thetas = best_thetas


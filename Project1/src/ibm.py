import sys
import os
import numpy as np
import math
import gc

import time

class IBM:

  null_word = 0

  def __init__(self, e_vocab, f_vocab):
    start = time.time()
    self.e_vocab = self._init_vocab(e_vocab,True)
    #print 'initialized e_vocab', time.time() - start
    start = time.time()
    self.f_vocab = self._init_vocab(f_vocab,False)
    #print 'initialized f_vocab', time.time() - start
    sys.stdout.flush()

  def _init_vocab(self, set_vocab, add_null_word):
    vocab = dict()
    i = 1 if add_null_word else 0
    for word in set_vocab:
      vocab[word] = i
      i += 1
    return vocab

  def _define_parameters(self):
    # Return [a list of] all parameters
    return

  def _get_parameters(self, params, e_sentence_len, f_sentence_len, f_sentence_index, f_word_id):
    # Return [a list of] all parameters for f_word_id
    return

  def _uniform_initialize_parameter(self, params_f, e_sentence_index, e_word_id):
    # Uniformly initalize all joint parameters for f and e
    return

  def _define_expectations(self):
    # Return a list of of joint expectations and expectations for english words
    return

  def _conditional_probabilities(self, params_f_es, e_sentence):
    # Retrun the conditional probabilities of the french word, all params
    # for which are passed, given the null word and the words in the english sentence
    # NULL WORD IS EXPECTED TO BE LAST!
    return # [P(f|e),...,P(f|e),P(f|0)]

  def _update_expectations(self, expectations, e_len, i, e_word_id, f_len, j, f_word_id, update_value):
    # Update all possible expectations for f_word_id and e_word_id
    return

  def _initialize_parameters(self, english, french, init_type, ibm1):
    # Initializes parameters
    return

  def _uniform_initialize_parameters(self, english, french):
    params = self._define_parameters()
    for sentence in range(len(english)):
      e_sentence = english[sentence]
      f_sentence = french[sentence]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        params_f = self._get_parameters(params, e_len, f_len, j, self.f_vocab[f_word])
        # null word
        self._uniform_initialize_parameter(params_f, self.null_word, self.null_word)
        for i, e_word in enumerate(e_sentence):
          self._uniform_initialize_parameter(params_f, i+1, self.e_vocab[e_word])
    return params

  # I tried putting maps and getting rid of loops
  # It actually became 10x slower...
  # My guess is it was because of the overhead of creating lambda functions objects
  def _e_step(self, english, french):
    expectations = self._define_expectations()
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        f_word_id = self.f_vocab[f_word]
        params_f_es = self._get_parameters(self.params, e_len, f_len, j, f_word_id)
        conditional_probabilities = self._conditional_probabilities(params_f_es, e_sentence)
        # Compute "normalizing constant"
        norm_const = sum(conditional_probabilities)
        # null word
        update_value = conditional_probabilities[-1] / norm_const
        self._update_expectations(expectations, e_len, self.null_word, self.null_word, f_len, j, f_word_id, update_value)
        # normal words
        for i, e_word in enumerate(e_sentence):
          e_word_id = self.e_vocab[e_word]
          update_value = conditional_probabilities[i] / norm_const
          self._update_expectations(expectations, e_len, i+1, e_word_id, f_len, j, f_word_id, update_value)
    return expectations

  def _update_parameters(self, params, joint_expectations, expectations):
    for e in expectations:
      expectation_e = expectations[e]
      joint_expectations_e = joint_expectations[e]
      for f in joint_expectations_e:
        params[f][e] = joint_expectations_e[f] / expectation_e

  def _m_step(self, expectations):
    return

  def _iteration(self, english, french):
    gc.collect()
    # E-step
    start = time.time()
    expectations = self._e_step(english, french)
    #print 'E',time.time()-start
    # M-step
    start = time.time()
    self._m_step(expectations)
    #print 'M',time.time()-start

  def _compute_log_likelihood(self, english, french):
    log_likelihood = 0
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        f_word_id = self.f_vocab[f_word]
        params_f_es = self._get_parameters(self.params, e_len, f_len, j, f_word_id)
        probabilities = self._conditional_probabilities(params_f_es, e_sentence)
        # max seems to be significantly faster for lists than np.amax
        # or at least for lists of size <= 50
        marginalized = sum(probabilities)
        assert(marginalized > 0)
        log_likelihood += math.log(marginalized)
    return log_likelihood

  def _compute_AER(self, alignments, sure, possible):
    probable = sure.union(possible)
    correct_sure = len(alignments.intersection(sure))
    correct_probable = len(alignments.intersection(probable))
    aer = 1 - float(correct_sure + correct_probable) / float(len(alignments) + len(sure))
    return aer

  def _split_data(self, english, french):
    sentence_count = len(english)
    sentence_nos = np.arange(sentence_count)
    #shuffles in-place
    np.random.shuffle(sentence_nos)
    #take 80% of data as train data, the rest is for validation
    training_count = math.floor(0.8 * sentence_count)
    training = sentence_nos[:training_count]
    validation = sentence_nos[training_count:]
    return training, validation

  def _write_alignments_to_file(self, f, alignments):
    for alignment in alignments:
      al_string = '%04d' % alignment[0]
      al_string += ' '
      al_string += str(alignment[1])
      al_string += ' '
      al_string += str(alignment[2])
      al_string += '\n'
      f.write(al_string)

  # List of sentences
  # sentence - list of words
  # ibm1 - path for file with serialized ibm1
  def train(self, english, french, iterations, test_data, model_name, init_type = 'uniform', ibm1 = ''):
    assert(len(english) == len(french))
    # Initialize params
    start = time.time()
    self.params = self._initialize_parameters(english, french, init_type, ibm1)
    #print 'initialized params', time.time() - start
    sys.stdout.flush()

    # save best params after every iteration
    best_params = self.params
    best_log_likelihood = float('-inf')
    best_aer = 1

    alignments_dir_name = model_name
    if -1 != model_name.find('.'):
      alignments_dir_name = model_name[0:model_name.find('.')]
    if not os.path.exists(alignments_dir_name):
      os.makedirs(alignments_dir_name)

    # Run EM
    for i in range(iterations):
      i_start = time.time()

      #possibly split the data into train and validation data for cross-validation?
      #training, validation = self._split_data(english, french) #training, validation - arrays of indices of sentences
      start = time.time()
      self._iteration(english, french)
      #print 'iteration finished', time.time() - start
      sys.stdout.flush()

      start = time.time()
      log_likelihood = self._compute_log_likelihood(english, french)
      #print 'computed log-likelihood', time.time() - start

      start = time.time()
      # get alignments for test data
      alignments = self.get_alignments(test_data[0], test_data[1])
      #print 'got alignments', time.time() - start

      start = time.time()
      fname = alignments_dir_name + '/' + str(i) + '.txt'
      f = open(fname, 'w')
      self._write_alignments_to_file(f, alignments)
      f.close()
      #print 'wrote alignments to file', time.time() - start

      start = time.time()
      # testing with the same alignments until we get anotated data
      aer = self._compute_AER(alignments, test_data[2], test_data[3])
      #print 'computed AER', time.time() - start

      if best_aer > aer:
        best_aer = aer

      start = time.time()
      if best_log_likelihood < log_likelihood:
        best_params = self.params
        best_log_likelihood = log_likelihood

      #print 'Iteration', i, 'Time:',time.time() - i_start,'s','Log-likelihood',log_likelihood,'AER',aer
      sys.stdout.flush()
    
    #recover best params
    self.params = best_params
    return best_aer


import numpy as np
import math
import gc

import time

class IBM:

  null_word = 0

  def __init__(self, e_vocab, f_vocab):
    start = time.time()
    self.e_vocab = self._init_vocab(e_vocab,True)
    print 'initialized e_vocab', time.time() - start
    start = time.time()
    self.f_vocab = self._init_vocab(f_vocab,False)
    print 'initialized f_vocab', time.time() - start

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

  def _random_initialize_parameter(self, params_f, e_sentence_index, e_word_id):
    # Random initalize all joint parameters for f and e
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

  def _random_initialize_parameters(self, english, french):
    params = self._define_parameters()
    for sentence in range(len(english)):
      e_sentence = english[sentence]
      f_sentence = french[sentence]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        params_f = self._get_parameters(params, e_len, f_len, j, self.f_vocab[f_word])
        # null word
        self._random_initialize_parameter(params_f, self.null_word, self.null_word)
        for i, e_word in enumerate(e_sentence):
          self._random_initialize_parameter(params_f, i+1, self.e_vocab[e_word])
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
    print 'E',time.time()-start
    # M-step
    start = time.time()
    self._m_step(expectations)
    print 'M',time.time()-start

  def _compute_log_like_and_alignments(self, english, french):
    log_likelihood = 0
    alignments = []
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      sentence_log_likelihood = 0
      s_alignments = set()
      for j, f_word in enumerate(f_sentence):
        f_word_id = self.f_vocab[f_word]
        params_f_es = self._get_parameters(self.params, e_len, f_len, j, f_word_id)
        probabilities = self._conditional_probabilities(params_f_es, e_sentence)
        # max seems to be significantly faster for lists than np.amax
        # or at least for lists of size <= 50
        max_probability = max(probabilities)
        alignment = probabilities.index(max_probability)
        # ignore null word
        if alignment < len(e_sentence):
          # let indexing start from 1 (at least until we get anotated data)
          s_alignments.add( (j+1, alignment+1) )
        assert(max_probability > 0)
        sentence_log_likelihood += math.log(max_probability)
      log_likelihood += sentence_log_likelihood
      alignments.append(s_alignments)
    return log_likelihood, alignments

  def _compute_AER(self, alignments, sure, possible):
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

  # List of sentences
  # sentence - list of words
  def train(self, english, french, iterations):
    assert(len(english) == len(french))
    # Initialize params
    start = time.time()
    self.params = self._random_initialize_parameters(english, french)
    print 'initialized params', time.time() - start

    # save best params after every iteration
    best_params = self.params
    best_log_likelihood = float('-inf')

    # Run EM
    for i in range(iterations):
      i_start = time.time()

      #possibly split the data into train and validation data for cross-validation?
      #training, validation = self._split_data(english, french) #training, validation - arrays of indices of sentences
      start = time.time()
      self._iteration(english, french)
      print 'iteration finished', time.time() - start
      
      start = time.time()
      log_likelihood, alignments = self._compute_log_like_and_alignments(english, french)
      print 'computed log-likelihood', time.time() - start

      start = time.time()
      # testing with the same alignments until we get anotated data
      aer = self._compute_AER(alignments, alignments, alignments)
      print 'computed AER', time.time() - start
      
      start = time.time()
      if best_log_likelihood < log_likelihood:
        best_params = self.params
        best_log_likelihood = log_likelihood

      print 'Iteration', i, 'Time:',time.time() - i_start,'s','Log-likelihood',log_likelihood,'AER',aer
    
    #recover best params
    self.params = best_params


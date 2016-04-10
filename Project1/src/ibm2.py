import numpy as np
import random
import gc
from collections import defaultdict

import time

# Needed for the nested default dictionaries to be serializable
def default_dict():
  return defaultdict(float)

class IBM2:

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

  def __random_initialize_pars(self, english, french):
    # add 1 for the null word
    thetas = defaultdict(default_dict)
    qs = defaultdict(default_dict)
    for sentence in range(len(english)):
      e_sentence = english[sentence]
      f_sentence = french[sentence]
      for f_word in f_sentence:
        thetas_f = thetas[self.f_vocab[f_word]]
        qs_f = qs[self.f_vocav[f_word]]
        # null word
        thetas_f[self.null_word] = random.random()
        qs_f = qs[self.null_word] = random.random()
        for e_word in e_sentence:
          thetas_f[self.e_vocab[e_word]] = random.random()
          qs_f[self.e_vocab[e_word]] = random.random()
    return thetas, qs


  # I tried putting maps and getting rid of loops
  # It actually became 10x slower...
  # My guess is it was because of the overhead of creating lambda functions objects
  def __e_step(self, english, french):
    #a default dict of default dicts of floats
    joint_expectations_t = defaultdict(default_dict)
    expectations_t = defaultdict(float)
    joint_expectations_q = defaultdict(default_dict)
    expectations_q = defaultdict(float)
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      for f_word in f_sentence:
        f_index = self.f_vocab[f_word]
        theta_f_es = self.thetas[f_index]
        qs_f_es = self.qs[f_index] # i used q as the alignment parameter, following collins

        # Compute "normalizing constant"
        theta_f = theta_f_es[self.null_word]
        q_f = q_f_es[self.null_word]
        for e_word in e_sentence:
          theta_f += theta_f_es[self.e_vocab[e_word]]
          q_f += q_f_es[self.e_vocab[e_word]]

        # null word
        update_value = (theta_f_es[self.null_word] * q_f_es[self.null_word]) / (theta_f + q_f)
        joint_expectations_t[self.null_word][f_index] += update_value
        expectations_t[self.null_word] += update_value
        joint_expectations_q[self.null_word][f_index] += update_value
        expectations_q[self.null_word] += update_value

        # normal words
        for e_word in e_sentence:
          e_index = self.e_vocab[e_word]
          update_value = (theta_f_es[e_index] * q_f_es[e_index]) / (theta_f + q_f)
          joint_expectations_t[e_index][f_index] += update_value
          expectations_t[e_index] += update_value
          joint_expectations_q[e_index][f_index] += update_value
          expectations_q[e_index] += update_value

    return joint_expectations_t, expectations_t, joint_expectations_q, expectations_q

  def __m_step(self, joint_expectations_t, expectations_t, joint_expectations_e, expectations_e):
    for e in expectations_t:
      expectation_e_t = expectations_t[e]
      joint_expectations_e_t = joint_expectations_t[e]
      for f in joint_expectations_e_t:
        self.thetas[f][e] = joint_expectations_e_t[f] / expectation_e_t

    for e in expectations_q:
      expectation_e_q = expectations_q[e]
      joint_expectations_e_q = joint_expectations_q[e]
      for f in joint_expectations_e_q:
        self.qs[f][e] = joint_expectations_e_q[f] / expectation_e_q

  def __iteration(self, english, french):
    gc.collect()
    # E-step
    start = time.time()
    joint_expectations_t, expectations_t, joint_expectations_q, expectations_q = self.__e_step(english, french)
    print 'E',time.time()-start
    # M-step
    start = time.time()
    self.__m_step(joint_expectations_t, expectations_t, joint_expectations_q, expectations_q)
    print 'M',time.time()-start

  def __compute_log_likelihood(self, english, french):
    log_likelihood = 0
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      sentence_log_likelihood = 0
      for f_word in f_sentence:
        theta_f_es = self.thetas[self.f_vocab[f_word]]
        # Get probabilities for the words in the english sentence
        probabilities = map(lambda e_word: theta_f_es[self.e_vocab[e_word]], e_sentence)
        # Add null word
        probabilities.append(theta_f_es[self.null_word])
        # max seems to be significantly faster for lists than np.amax
        # or at least for lists of size <= 50
        max_probability = max(probabilities)
        assert(max_probability > 0)
        sentence_log_likelihood += np.log(max_probability)
      log_likelihood += sentence_log_likelihood
    return log_likelihood

  def __split_data(self, english, french):
    sentence_count = len(english)
    sentence_nos = np.arange(sentence_count)
    #shuffles in-place
    np.random.shuffle(sentence_nos)
    #take 80% of data as train data, the rest is for validation
    training_count = np.floor(0.8 * sentence_count)
    training = sentence_nos[:training_count]
    validation = sentence_nos[training_count:]
    return training, validation
    
  # List of sentences
  # sentence - list of words
  def train(self, english, french, iterations):
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
      log_likelihood = self.__compute_log_likelihood(english, french)
      print 'computed log-likelihood', time.time() - start
      
      start = time.time()
      if best_log_likelihood < log_likelihood:
        best_thetas = self.thetas
        best_log_likelihood = log_likelihood

      print 'Iteration', i, 'Time:',time.time() - i_start,'s','Log-likelihood',log_likelihood
    
    #recover best thetas
    self.thetas = best_thetas

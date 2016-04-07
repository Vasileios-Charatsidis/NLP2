from collections import defaultdict
import numpy as np

import time

class IBM1:

  #Define a sensible null word
  null_word = 'blah'

  def __init__(self, e_vocab, f_vocab):
    self.e_vocab = e_vocab
    self.f_vocab = f_vocab
    self.thetas = self.__random_initialize_thetas()

  def __random_initialize_thetas(self):
    thetas = defaultdict(lambda : defaultdict(float))
    for f_word in self.f_vocab:
      thetas_f = thetas[f_word]
      # null word
      thetas_f[self.null_word] = np.random.random(1)[0]
      # normal words
      for e_word in self.e_vocab:
        thetas_f[e_word] = np.random.random(1)[0]
    return thetas

  def __e_step(self, english, french):
    #a default dict of default dicts of floats
    joint_expectations = defaultdict(lambda : defaultdict(float))
    expectations  = defaultdict(float)
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      for f_word in f_sentence:
        theta_f_es = self.thetas[f_word]
        
        theta_f = theta_f_es[self.null_word]
        for e_word in e_sentence:
          theta_f += theta_f_es[e_word]
        
        # null word
        update_value = theta_f_es[self.null_word] / theta_f
        joint_expectations[self.null_word][f_word] += update_value
        expectations[self.null_word] += update_value
        # normal words
        for e_word in e_sentence:
          update_value = theta_f_es[e_word] / theta_f
          joint_expectations[e_word][f_word] += update_value
          expectations[e_word] += update_value
          
    return joint_expectations, expectations

  def __m_step(self, joint_expectations, expectations):
    # null word
    expectation_e = expectations[self.null_word]
    joint_expectations_e = joint_expectations[self.null_word]
    for f_word in self.f_vocab:
      self.thetas[f_word][self.null_word] = joint_expectations_e[f_word] / expectation_e
    # normal words
    for e_word in self.e_vocab:
      expectation_e = expectations[e_word]
      joint_expectations_e = joint_expectations[e_word]
      for f_word in self.f_vocab:
        self.thetas[f_word][e_word] = joint_expectations_e[f_word] / expectation_e

  def __iteration(self, english, french):
    # E-step
    joint_expectations, expectations = self.__e_step(english, french)
    # M-step
    self.__m_step(joint_expectations, expectations)

  def __compute_log_likelihood(self, english, french):
    log_likelihood = 0
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      sentence_log_likelihood = 0
      for f_word in f_sentence:
        theta_f_es = self.thetas[f_word]
        # null word
        max_theta = np.log(theta_f_es[self.null_word])
        # normal words
        for e_word in e_sentence:
          if max_theta < np.log(theta_f_es[e_word]):
            max_theta = np.log(theta_f_es[e_word])
        sentence_log_likelihood += max_theta
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
    # set expectation counts to 0?
    self.thetas = self.__random_initialize_thetas()

    for i in range(iterations):
      start = time.time()

      #possibly split the data into train and validation data for cross-validation?
      #training, validation = self.__split_data(english, french) #training, validation - arrays of indices of sentences
      self.__iteration(english, french)
      log_likelihood = self.__compute_log_likelihood(english, french)

      print 'Iteration', i, 'Time:',time.time() - start,'s','Log-likelihood',log_likelihood

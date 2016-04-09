import numpy as np
import gc

import time

class IBM1:

  #Define a sensible null word
  null_word = 0

  def __init__(self, e_vocab, f_vocab):
    start = time.time()
    self.e_vocab = self.__init_vocab(e_vocab,True)
    print 'initialized e_vocab', time.time() - start
    start = time.time()
    self.f_vocab = self.__init_vocab(f_vocab,False)
    print 'initialized f_vocab', time.time() - start
    start = time.time()
    self.thetas = self.__random_initialize_thetas()
    print 'initialized thetas', time.time() - start


  def __init_vocab(self, set_vocab, add_null_word):
    vocab = dict()
    i = 1 if add_null_word else 0
    for word in set_vocab:
      vocab[word] = i
      i += 1
    return vocab

  def __random_initialize_thetas(self):
    # add 1 for the null word
    thetas = np.zeros([len(self.f_vocab),1+len(self.e_vocab)],dtype='float16')
    thetas.fill(0.5)
#    for f_word in range(thetas.shape[0]):
#      start = time.time()
#      thetas_f = thetas[f_word]
#      for e_word in range(thetas.shape[1]):
#        thetas_f[e_word] = np.random.random(1)[0]
#      print f_word, time.time()-start
    return thetas

  def __e_step(self, english, french):
    #a default dict of default dicts of floats
    joint_expectations = np.zeros([1+len(self.e_vocab),len(self.f_vocab)],dtype='float16')
    expectations = np.zeros(1+len(self.e_vocab),dtype='float16')
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      for f_word in f_sentence:
        f_index = self.f_vocab[f_word]
        theta_f_es = self.thetas[f_index]
        
        theta_f = theta_f_es[self.null_word]
        for e_word in e_sentence:
          theta_f += theta_f_es[self.e_vocab[e_word]]
        
        # null word
        update_value = theta_f_es[self.null_word] / theta_f
        joint_expectations[self.null_word][f_index] += update_value
        expectations[self.null_word] += update_value
        # normal words
        for e_word in e_sentence:
          e_index = self.e_vocab[e_word]
          update_value = theta_f_es[e_index] / theta_f
          joint_expectations[e_index][f_index] += update_value
          expectations[e_index] += update_value
          
    return joint_expectations, expectations

  def __m_step(self, joint_expectations, expectations):
    self.thetas = self.thetas.T
    for e in range(self.thetas.shape[0]):
      expectation_e = expectations[e]
      joint_expectations_e = joint_expectations[e]
      self.thetas[e] = joint_expectations_e / expectation_e
    self.thetas = self.thetas.T

  def __iteration(self, english, french):
    gc.collect()
    # E-step
    start = time.time()
    joint_expectations, expectations = self.__e_step(english, french)
    print 'E',time.time()-start
    # M-step
    start = time.time()
    self.__m_step(joint_expectations, expectations)
    print 'M',time.time()-start

  def __compute_log_likelihood(self, english, french):
    log_likelihood = 0
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      sentence_log_likelihood = 0
      for f_word in f_sentence:
        theta_f_es = self.thetas[self.f_vocab[f_word]]
        # null word
        max_theta = np.log(theta_f_es[self.null_word])
        # normal words
        for e_word in e_sentence:
          e_index = self.e_vocab[e_word]
          if max_theta < np.log(theta_f_es[e_index]):
            max_theta = np.log(theta_f_es[e_index])
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
    # save best thetas after every iteration
    best_thetas = self.thetas
    best_log_likelihood = float('-inf')
    for i in range(iterations):
      start = time.time()

      #possibly split the data into train and validation data for cross-validation?
      #training, validation = self.__split_data(english, french) #training, validation - arrays of indices of sentences
      self.__iteration(english, french)
      print 'iteration finished', time.time() - start
      
      start = time.time()
      log_likelihood = self.__compute_log_likelihood(english, french)
      print 'computed log-likelihood', time.time() - start
      
      start = time.time()
      if best_log_likelihood < log_likelihood:
        best_thetas = self.thetas
        best_log_likelihood = log_likelihood

      print 'Iteration', i, 'Time:',time.time() - start,'s','Log-likelihood',log_likelihood
    
    #recover best thetas
    self.thetas = best_thetas

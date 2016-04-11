import random
import gc
from collections import defaultdict

from ibm import IBM

import time

# Needed for the nested default dictionaries to be serializable
def default_dict():
  return defaultdict(float)

class IBM1(IBM):

  def __random_initialize_thetas(self, english, french):
    thetas = defaultdict(default_dict)
    for sentence in range(len(english)):
      e_sentence = english[sentence]
      f_sentence = french[sentence]
      for f_word in f_sentence:
        thetas_f = thetas[self.f_vocab[f_word]]
        # null word
        thetas_f[self.null_word] = random.random()
        for e_word in e_sentence:
          thetas_f[self.e_vocab[e_word]] = random.random()
    return thetas

  # I tried putting maps and getting rid of loops
  # It actually became 10x slower...
  # My guess is it was because of the overhead of creating lambda functions objects
  def __e_step(self, english, french):
    #a default dict of default dicts of floats
    joint_expectations = defaultdict(default_dict)
    expectations = defaultdict(float)
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      for f_word in f_sentence:
        f_index = self.f_vocab[f_word]
        theta_f_es = self.thetas[f_index]
        # Compute "normalizing constant"
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
    for e in expectations:
      expectation_e = expectations[e]
      joint_expectations_e = joint_expectations[e]
      for f in joint_expectations_e:
        self.thetas[f][e] = joint_expectations_e[f] / expectation_e

  def __compute_log_like_and_alignments(self, english, french):
    log_likelihood = 0
    alignments = []
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      sentence_log_likelihood = 0
      s_alignments = set()
      for j, f_word in enumerate(f_sentence):
        theta_f_es = self.thetas[self.f_vocab[f_word]]
        # Get probabilities for the words in the english sentence
        probabilities = map(lambda e_word: theta_f_es[self.e_vocab[e_word]], e_sentence)
        # Add null word
        probabilities.append(theta_f_es[self.null_word])
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


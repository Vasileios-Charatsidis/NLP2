import random
import gc
from collections import defaultdict

from ibm import IBM

import time

# Needed for the nested default dictionaries to be serializable
def default_dict():
  return defaultdict(float)

class IBM2(IBM):

  def __random_initialize_pars(self, english, french):
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

        probabilities = [theta_f_es[self.null_word] * q_f_es[self.null_word]]
        for e_word in e_sentence:
          probabilities.append(theta_f_es[self.e_vocab[e_word]] * q_f_es[self.e_vocab[e_word]])
        # Compute "normalizing constant"
        norm_const = sum(probabilities)

        # null word
        update_value = (theta_f_es[self.null_word] * q_f_es[self.null_word]) / (norm_const)
        joint_expectations_t[self.null_word][f_index] += update_value
        expectations_t[self.null_word] += update_value
        joint_expectations_q[self.null_word][f_index] += update_value
        expectations_q[self.null_word] += update_value
        # normal words
        for i, e_word in enumerate(e_sentence):
          e_index = self.e_vocab[e_word]
          update_value = probabilities[i+1] / norm_const
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
        qs_f_es = self.qs[self.f_vocab[f_word]]
        # Get probabilities for the words in the english sentence
        probabilities = map(lambda e_word: theta_f_es[self.e_vocab[e_word]] * qs_f_es[self.e_vocab[e_word]], e_sentence)
        # Add null word
        probabilities.append(theta_f_es[self.null_word] * qs_f_es[self.null_word])
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
    joint_expectations_t, expectations_t, joint_expectations_q, expectations_q = self.__e_step(english, french)
    print 'E',time.time()-start
    # M-step
    start = time.time()
    self.__m_step(joint_expectations_t, expectations_t, joint_expectations_q, expectations_q)
    print 'M',time.time()-start


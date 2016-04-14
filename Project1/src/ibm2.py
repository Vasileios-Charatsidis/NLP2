import random
import sys
from collections import defaultdict
import copy
import cPickle as pickle

from ibm import IBM

# Needed for the nested default dictionaries to be serializable
def default_dict1():
  return defaultdict(float)

def default_dict2():
  return defaultdict(default_dict1)

def default_dict3():
  return defaultdict(default_dict2)

class IBM2(IBM):

  def _define_parameters(self):
    # make an iterable of
    # 0. translation probabilities
    # 1. alignment probabilities
    return [default_dict2(),defaultdict(default_dict3)]

  def _get_parameters(self, params, e_sentence_len, f_sentence_len, f_sentence_index, f_word_id):
    # translation probabilities are in param[0]
    # alignment probabilities are in param[1]
    # get the ones for the given french word ID and
    # Return them in an iterable
    return [params[0][f_word_id], params[1][e_sentence_len][f_sentence_len][f_sentence_index]]

  def _uniform_initialize_parameter(self, params, e_sentence_index, e_word_id):
    uniform_prob = 0.5
    # translation probabilities
    params[0][e_word_id] = uniform_prob
    # alignment probabilities
    params[1][e_sentence_index] = uniform_prob

  def _random_initialize_parameter(self, params, e_sentence_index, e_word_id):
    # translation probabilities
    params[0][e_word_id] = random.random()
    # alignment probabilities
    params[1][e_sentence_index] = random.random()

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

  def _initialize_from_ibm1(self, english, french, ibm1_file_name):
    ibm1_file = open(ibm1_file_name, 'rb')
    ibm1_model = pickle.load(ibm1_file)
    ibm1_file.close()

    params = self._define_parameters()
    params[0] = copy.deepcopy(ibm1_model.params)
    for sentence in range(len(english)):
      e_sentence = english[sentence]
      f_sentence = french[sentence]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        params_f = self._get_parameters(params, e_len, f_len, j, self.f_vocab[f_word])
        params_f[1][self.null_word] = random.random()
        for i, e_word in enumerate(e_sentence):
          params_f[1][i+1] = random.random()
    return params

  def _initialize_parameters(self, english, french, init_type, ibm1):
    params = None
    if 'uniform' == init_type.lower():
      params = self._uniform_initialize_parameters(english, french)
    elif 'random' == init_type.lower():
      params = self._random_initialize_parameters(english, french)
    elif 'ibm1' == init_type.lower():
      params = self._initialize_from_ibm1(english, french, ibm1)
    else:
      print 'No such initialization option. Exiting'
      sys.exit()
    return params

  def _define_expectations(self):
    # Return tuple of joint expectations of translations and
    # expectations for translations of english words
    # joint transl. expectations, transl. expectations, joint align. expectations, align. expectations
    return [default_dict2(), default_dict1(), defaultdict(default_dict3), default_dict3()]

  def _conditional_probabilities(self, params_f_es, e_sentence):
    # Probability given the NULL WORD is LAST
    probs = map(lambda (i, e_word): params_f_es[0][self.e_vocab[e_word]] * params_f_es[1][i+1], enumerate(e_sentence))
    # add null word at back
    probs.append(params_f_es[0][self.null_word] * params_f_es[1][self.null_word])
    return probs

  def _update_expectations(self, expectations, e_len, i, e_word_id, f_len, j, f_word_id, update_value):
    # 0 - joint transl. expectations on e first and then f
    expectations[0][e_word_id][f_word_id] += update_value
    # 1 - transl. expectations for the english words only
    expectations[1][e_word_id] += update_value
    # 2 - joint align. expectations on e first and then f
    expectations[2][e_len][f_len][i][j] += update_value
    # 3 - align. expectations for the english words only
    expectations[3][e_len][f_len][i] += update_value

  def _m_step(self, expectations):
    self._update_parameters(self.params[0], expectations[0], expectations[1])
    joint = expectations[2]
    marginal = expectations[3]
    for e_len in joint:
      joint_e = joint[e_len]
      marginal_e = marginal[e_len]
      params_e = self.params[1][e_len]
      for f_len in joint_e:
        self._update_parameters(params_e[f_len], joint_e[f_len], marginal_e[f_len])

  def get_alignments(self, english, french):
    alignments = set()
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        probabilities = []
        if f_word in self.f_vocab:
          f_word_id = self.f_vocab[f_word]
          params_f_es = self._get_parameters(self.params, e_len, f_len, j, f_word_id)
          # Compute conditional probabilities
          for i, e_word in enumerate(e_sentence):
            if e_word in self.e_vocab:
              probabilities.append(params_f_es[0][self.e_vocab[e_word]] * params_f_es[1][i+1])
            else:
              probabilities.append(1e-6 * params_f_es[1][i+1])
          probabilities.append(params_f_es[0][self.null_word] * params_f_es[1][self.null_word])
        else:
          # Use only alignment probabilitites as we do not have translation ones
          params_f_es = self.params[1][e_len][f_len][j]
          # Compute conditional probabilities
          probabilities = map(lambda (i,_): params_f_es[i+1], enumerate(e_sentence))
          probabilities.append(params_f_es[self.null_word])
        # max seems to be significantly faster for lists than np.amax
        # or at least for lists of size <= 50
        max_probability = max(probabilities)
        alignment = probabilities.index(max_probability)
        # ignore null word
        if alignment < len(e_sentence):
          # let indexing start from 1 (at least until we get anotated data)
          alignments.add( (sentence_no+1, j+1, alignment+1) )
    return alignments

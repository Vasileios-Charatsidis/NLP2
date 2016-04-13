import random
from collections import defaultdict

from ibm import IBM

# Needed for the nested default dictionaries to be serializable
def default_dict():
  return defaultdict(float)

class IBM1(IBM):

  def _define_parameters(self):
    # make an iterable of translation probabilities and nothing else
    return defaultdict(default_dict)

  def _get_parameters(self, params, e_sentence_len, f_sentence_len, f_sentence_index, word_id):
    # translation probabilities are in param[0]
    # get the ones for the given word ID and
    # Return them in an iterable of them and nothing else
    return params[word_id]

  def _random_initialize_parameter(self, params, e_sentence_index, e_word_id):
    params[e_word_id] = random.random()

  def _initialize_parameters(self, english, french, init_type, ibm1):
    return self._random_initialize_parameters(english, french)

  def _define_expectations(self):
    # Return list of joint expectations of translations and
    # expectations for translations of english words
    # joint trans. expectations, transl. expectations
    return [defaultdict(default_dict), defaultdict(float)]

  def _conditional_probabilities(self, params_f_es, e_sentence):
    # Probability given the NULL WORD is LAST
    probs = map(lambda e_word: params_f_es[self.e_vocab[e_word]], e_sentence)
    # add null word at back
    probs.append(params_f_es[self.null_word])
    return probs

  def _update_expectations(self, expectations, e_len, i, e_word_id, f_len, j, f_word_id, update_value):
    # 0 - joint translation expectations on e first and then f
    expectations[0][e_word_id][f_word_id] += update_value
    # 1 - translation expectations for the english words only
    expectations[1][e_word_id] += update_value

  def _m_step(self, expectations):
    self._update_parameters(self.params, expectations[0], expectations[1])

  def get_alignments(self, english, french):
    alignments = set()
    for sentence_no in range(len(english)):
      e_sentence = english[sentence_no]
      f_sentence = french[sentence_no]
      e_len = len(e_sentence)
      f_len = len(f_sentence)
      for j, f_word in enumerate(f_sentence):
        if f_word not in self.f_vocab:
          continue
        f_word_id = self.f_vocab[f_word]
        params_f_es = self._get_parameters(self.params, e_len, f_len, j, f_word_id)
        # Compute conditional probabilities
        probabilities = map(lambda e_word: params_f_es[self.e_vocab[e_word]] if e_word in self.e_vocab else 1e-6, e_sentence)
        # add null word at back
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


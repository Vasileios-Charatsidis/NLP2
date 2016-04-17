from ibm1 import IBM1

class IBM1_SMOOTH(IBM1):

  V = 40000

  def __init__(self, e_vocab, f_vocab, n = 1):
    IBM1.__init__(self, e_vocab, f_vocab)
    self.n = n

  def _update_parameters(self, params, joint_expectations, expectations):
    for e in expectations:
      expectation_e = expectations[e]
      joint_expectations_e = joint_expectations[e]
      for f in joint_expectations_e:
        params[f][e] = (joint_expectations_e[f] + self.n) / (expectation_e + self.n * self.V)

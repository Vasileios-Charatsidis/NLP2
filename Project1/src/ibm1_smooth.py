from ibm import IBM1

class IBM1_SMOOTH(IBM1):

  def _update_parameters(self, params, joint_expectations, expectations, n, V):
    for e in expectations:
      expectation_e = expectations[e]
      joint_expectations_e = joint_expectations[e]
      for f in joint_expectations_e:
        params[f][e] = (joint_expectations_e[f] + n) / (expectation_e + n * V)



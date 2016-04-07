import sys

import dataloader as dl
from ibm1 import IBM1 as ibm1

#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print 'Usage: main english_file french_file iterations'
    sys.exit()
  
  english_fname = sys.argv[1]
  french_fname = sys.argv[2]
  iterations = int(sys.argv[3])
  
  # read data
  english, french, e_vocab, f_vocab = dl.read_data(english_fname, french_fname)
  # create a model object
  model = ibm1(e_vocab, f_vocab)
  # train model
  model.train(english, french, iterations)

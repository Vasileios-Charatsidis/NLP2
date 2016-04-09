import sys
import pickle
import gc

import dataloader as dl
from ibm1 import IBM1 as ibm1

import time


#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print 'Usage: python main.py english_file french_file iterations'
    sys.exit()
  
  english_fname = sys.argv[1]
  french_fname = sys.argv[2]
  iterations = int(sys.argv[3])
  
  # read data
  english, french, e_vocab, f_vocab = dl.read_data(english_fname, french_fname)
  del english_fname, french_fname
  
  print 'data read'
  print len(e_vocab), len(f_vocab)
  
  # create a model object
  model = ibm1(e_vocab, f_vocab)
  del e_vocab, f_vocab
  
  # train model
  print 'start training'
  model.train(english, french, iterations)
  del english
  del french
  gc.collect()
  
  # serialize model
  model_file = open('ibm1.pkl', 'wb')
  pickle.dump(model.thetas, model_file)
  model_file.close()

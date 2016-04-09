import sys
import cPickle as pickle
import gc

import dataloader as dl
from ibm1 import IBM1 as ibm1

import time


def delete_content(f):
  f.seek(0)
  f.truncate()

#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print 'Usage: python main.py english_file french_file iterations'
    sys.exit()
  
  english_fname = sys.argv[1]
  french_fname = sys.argv[2]
  iterations = int(sys.argv[3])
  
  # read data
  start = time.time()
  english, french, e_vocab, f_vocab = dl.read_data(english_fname, french_fname)
  del english_fname, french_fname
  gc.collect()
  
  print 'data read', time.time() - start
  print 'English vocab size:', len(e_vocab)
  print 'French vocab size:', len(f_vocab)
  
  # create a model object
  model = ibm1(e_vocab, f_vocab)
  del e_vocab, f_vocab
  gc.collect()
  
  # train model
  print 'start training'
  model.train(english, french, iterations)
  print 'finished training'

  # clean up some memory
  del english
  del french
  gc.collect()
  
  print 'start serialization'
  start = time.time()
  # serialize model
  model_file = open('ibm1.pkl', 'wb')
  delete_content(model_file)
  pickle.dump(model, model_file)
  model_file.close()
  print 'finished serialization', time.time() - start

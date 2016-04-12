import sys
import cPickle as pickle
import gc

import dataloader as dl
from ibm1 import IBM1 as ibm1
from ibm2 import IBM2 as ibm2

import time


def _delete_content(f):
  f.seek(0)
  f.truncate()

def _create_modelibm1(model_type, e_vocab, f_vocab):
  if "ibm1" == model_type.lower():
    return ibm1(e_vocab, f_vocab)
  elif "ibm2" == model_type.lower():
    return ibm2(e_vocab, f_vocab)
  else:
    sys.stderr.write(model_type + 'is not a valid model type.\nExiting.')
    sys.exit()

#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 5:
    print 'Usage: python main.py model_type english_file french_file iterations'
    sys.exit()
  
  model_type = sys.argv[1]
  english_fname = sys.argv[2]
  french_fname = sys.argv[3]
  iterations = int(sys.argv[4])
  
  # read data
  start = time.time()
  english, french, e_vocab, f_vocab = dl.read_data(english_fname, french_fname)
  del english_fname, french_fname
  gc.collect()
  
  print 'data read', time.time() - start
  print 'English vocab size:', len(e_vocab)
  print 'French vocab size:', len(f_vocab)
  
  # create a model object
  model = _create_modelibm1(model_type, e_vocab, f_vocab)
  print isinstance(model, ibm1)
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
  _delete_content(model_file)
  pickle.dump(model, model_file)
  model_file.close()
  print 'finished serialization', time.time() - start

import sys
import cPickle as pickle
import gc

import dataloader as dl
from ibm1 import IBM1
from ibm1_add0 import IBM1_add0
from ibm1_smooth import IBM1_SMOOTH
from ibm2 import IBM2

import time

error = 'Usage: python main.py model_type train_english train_french test_english test_french alignments iterations model_name [init_type ibm1_file]\n\
        init_type is one of uniform, random, ibm1'

def _delete_content(f):
  f.seek(0)
  f.truncate()

def _create_model(model_type, e_vocab, f_vocab):
  if "ibm1" == model_type.lower():
    return IBM1(e_vocab, f_vocab)
  elif "ibm1_add0" == model_type.lower():
    return IBM1_add0(e_vocab, f_vocab)
  elif "ibm1_smooth" == model_type.lower():
    return IBM1_SMOOTH(e_vocab, f_vocab)
  elif "ibm2" == model_type.lower():
    return IBM2(e_vocab, f_vocab)
  else:
    sys.stderr.write(model_type + 'is not a valid model type.\nExiting.')
    sys.exit()

#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 9:
    print error
    sys.exit()
  
  model_type = sys.argv[1]
  train_english_fname = sys.argv[2]
  train_french_fname = sys.argv[3]
  test_english_fname = sys.argv[4]
  test_french_fname = sys.argv[5]
  alignments_fname = sys.argv[6]
  iterations = int(sys.argv[7])
  model_name = sys.argv[8]
  init_type = 'random'
  if len(sys.argv) > 9:
    init_type = sys.argv[9]
  ibm1 = ''
  if len(sys.argv) > 10:
    ibm1 = sys.argv[10]
  
  # read data
  start = time.time()
  english, french, e_vocab, f_vocab = dl.read_train_data(train_english_fname, train_french_fname)
  test_data = dl.read_test_data(test_english_fname, test_french_fname, alignments_fname)
  del train_english_fname, train_french_fname
  del test_english_fname, test_french_fname
  gc.collect()
  
  print 'data read', time.time() - start
  print 'English vocab size:', len(e_vocab)
  print 'French vocab size:', len(f_vocab)
  sys.stdout.flush()
  
  # create a model object
  model = _create_model(model_type, e_vocab, f_vocab)
  del e_vocab, f_vocab
  gc.collect()
  
  # train model
  print 'start training'
  # english, french, iterations, test_data, init_type = 'random', ibm1 = ''
  model.train(english, french, iterations, test_data, model_name, init_type, ibm1)
  print 'finished training'
  sys.stdout.flush()

  # clean up some memory
  del english
  del french
  gc.collect()
  
  print 'start serialization'
  start = time.time()
  # serialize model
  model_file = open(model_name, 'wb')
  _delete_content(model_file)
  pickle.dump(model, model_file)
  model_file.close()
  print 'finished serialization', time.time() - start

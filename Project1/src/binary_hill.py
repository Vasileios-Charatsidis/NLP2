import sys
import threading
import cPickle as pickle
import gc

import dataloader as dl
from ibm1 import IBM1
from ibm1_add0 import IBM1_add0
from ibm1_smooth import IBM1_SMOOTH
from ibm2 import IBM2

import time

error = 'Usage: python binary_hill.py model_type train_english train_french test_english test_french alignments iterations model_name'

class runThread(threading.Thread):

  def __init__(self, model_type, e_vocab, f_vocab, hyperparam, english, french, iterations, test_data, model_name, aers, i):
    threading.Thread.__init__(self)
    self.model_type = model_type
    self.e_vocab = e_vocab
    self.f_vocab = f_vocab
    self.hyperparam = hyperparam
    self.english = english
    self.french = french
    self.iterations = iterations
    self.test_data = test_data
    self.model_name = model_name
    self.aers = aers
    self.i = i

  def run(self):
    _climb_once(self.model_type, self.e_vocab, self.f_vocab, self.hyperparam, self.english, self.french, self.iterations, self.test_data, self.model_name, self.aers, self.i)

def _delete_content(f):
  f.seek(0)
  f.truncate()

def _create_model(model_type, e_vocab, f_vocab, hyperparam):
  if "ibm1_add0" == model_type.lower():
    return IBM1_add0(e_vocab, f_vocab, hyperparam)
  elif "ibm1_smooth" == model_type.lower():
    return IBM1_SMOOTH(e_vocab, f_vocab, hyperparam)
  else:
    sys.stderr.write(model_type + 'is not a valid model type.\nExiting.')
    sys.exit()

def _climb_once(model_type, e_vocab, f_vocab, hyperparam, english, french, iterations, test_data, model_name, aers, i):
  # create a model object
  model = _create_model(model_type, e_vocab, f_vocab, hyperparam)
  del e_vocab, f_vocab
  gc.collect()
  
  # train model
  print 'start training'
  # english, french, iterations, test_data, init_type = 'random', ibm1 = ''
  aers[i] = model.train(english, french, iterations, test_data, model_name)
  print 'finished training'
  sys.stdout.flush()

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
  
  lcr_aer = [0, 0]
  left = 1
  right = 100
  _climb_once(model_type, e_vocab, f_vocab, right, english, french, iterations, test_data, model_name, lcr_aer, 0)
  _climb_once(model_type, e_vocab, f_vocab, right, english, french, iterations, test_data, model_name, lcr_aer, 1)
  left_aer = lcr_aer[0]
  right_aer = lcr_aer[1]
  while left + 1 < right:
    current = int((left + right) / 2)
    new_left = int((left + current) / 2)
    new_right = int((current + right) / 2)
    threads = []
    lcr_aer = [0, 0, 0]
    threads.append(runThread(model_type, e_vocab, f_vocab, new_left, english, french, iterations, test_data, model_name+str(new_left), lcr_aer, 0))
    threads.append(runThread(model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 1))
    threads.append(runThread(model_type, e_vocab, f_vocab, new_right, english, french, iterations, test_data, model_name+str(new_right), lcr_aer, 2))
    # results should be passed as argOUT to _climb_once
    #p = Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, new_left, english, french, iterations, test_data, model_name))
    #p.join()
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    #print new_left_aer, current_aer, new_right_aer
    #prepare for next iteration
    #lcr_aer = [new_left_aer, current_aer, new_right_aer]
    lcr = [left, new_left, current, new_right, right]
    lcr_aer[:0] = [left_aer]
    lcr_aer.append(right_aer)
    print lcr
    print lcr_aer
    
    left_aer = min(lcr_aer)
    li = lcr_aer.index(left_aer)
    left = lcr[li]
    lcr.remove(left)
    lcr_aer.remove(left_aer)
    
    right_aer = min(lcr_aer)
    ri = lcr_aer.index(right_aer)
    right = lcr[ri]
    
    if ri < li:
      tmp = left
      left = right
      right = tmp
      tmp = left_aer
      left_aer = right_aer
      right_aer = tmp
    
    print left, right
    print left_aer, right_aer

  if left_aer < right_aer:
    print left, left_aer
  else:
    print right, right_aer

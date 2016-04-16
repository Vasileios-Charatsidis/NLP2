import sys
import threading
from multiprocessing import Process, Manager
import cPickle as pickle
import gc

import dataloader as dl
from ibm1 import IBM1
from ibm1_add0 import IBM1_add0
from ibm1_smooth import IBM1_SMOOTH
from ibm2 import IBM2

import time

TIME = 300

error = 'Usage: python binary_hill.py model_type train_english train_french test_english test_french alignments iterations model_name max_hyperparam'

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
  if 0 != aers[i]:
    print hyperparam, 'already computed', aers[i]
    sys.stdout.flush()
    return

  print 'computing aer for', hyperparam
  sys.stdout.flush()

  # create a model object
  stdout = sys.stdout
  log_file = open(model_name+'.log','w')
  sys.stdout = log_file
  model = _create_model(model_type, e_vocab, f_vocab, hyperparam)
  del e_vocab, f_vocab
  gc.collect()
  
  # train model
  print 'start training'
  # english, french, iterations, test_data, init_type = 'random', ibm1 = ''
  aers[i] = model.train(english, french, iterations, test_data, model_name)
  print 'finished training'
  sys.stdout.flush()
  
  print 'start serialization'
  start = time.time()
  # serialize model
  model_file = open(model_name + '.pkl', 'wb')
  _delete_content(model_file)
  pickle.dump(model, model_file)
  model_file.close()
  print 'finished serialization', time.time() - start
  
  log_file.close()
  sys.stdout = stdout

#TODO think of a more sensible name for this script

if __name__ == '__main__':
  if len(sys.argv) < 10:
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
  right = int(sys.argv[9])
  
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
  # right is set from args
  current = int((left + right) / 2)

  #lcr_aer = [0, 0, 0]
  #_climb_once(model_type, e_vocab, f_vocab, left, english, french, iterations, test_data, model_name+str(left), lcr_aer, 0)
  #_climb_once(model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 1)
  #_climb_once(model_type, e_vocab, f_vocab, right, english, french, iterations, test_data, model_name+str(right), lcr_aer, 2)
  # Python has GIL (global interpreter locks) which actually make the multithreaded version run slower
  # If it were processes and not threads it might have gotten faster, but I am not willing to risk running out of memory
  
  #threads = []
  #threads.append(runThread(model_type, e_vocab, f_vocab, left, english, french, iterations, test_data, model_name+str(left), lcr_aer, 0))
  #threads.append(runThread(model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 1))
  #threads.append(runThread(model_type, e_vocab, f_vocab, right, english, french, iterations, test_data, model_name+str(right), lcr_aer, 2))
  #for thread in threads:
  # thread.start()
  #for thread in threads:
  #  thread.join()
  
  manager = Manager()
  
  lcr_aer = manager.list([0, 0, 0])
  processes = []
  processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, left, english, french, iterations, test_data, model_name+str(left), lcr_aer, 0)))
  processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 1)))
  processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, right, english, french, iterations, test_data, model_name+str(right), lcr_aer, 2)))
  for p in processes:
    p.start()
  for p in processes:
    p.join()

  # Let cores cool off
  time.sleep(TIME)

  left_aer = lcr_aer[0]
  current_aer = lcr_aer[1]
  right_aer = lcr_aer[2]
  
  new_left = int((left + current) / 2)
  new_right = int((current + right) / 2)
  # ok since it will always be ran with right >= 5
  lcr_aer = manager.list([left_aer, 0, current_aer, 0, right_aer])

  old_left = -1
  old_right = -1
  while (left + 1 < right) and (old_left != left or old_right != right):
    old_left = left
    old_right = right
    
    #lcr_aer = [left_aer, 0, current_aer, 0, right_aer]
    #_climb_once(model_type, e_vocab, f_vocab, new_left, english, french, iterations, test_data, model_name+str(new_left), lcr_aer, 1)
    #if 0 == current_aer:
    #  _climb_once(model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 2)
    #_climb_once(model_type, e_vocab, f_vocab, new_right, english, french, iterations, test_data, model_name+str(new_right), lcr_aer, 3)
    
    #threads = []
    #threads.append(runThread(model_type, e_vocab, f_vocab, new_left, english, french, iterations, test_data, model_name+str(new_left), lcr_aer, 1))
    #threads.append(runThread(model_type, e_vocab, f_vocab, new_right, english, french, iterations, test_data, model_name+str(new_right), lcr_aer, 3))
    #for thread in threads:
    #  thread.start()
    #for thread in threads:
    #  thread.join()
    
    print 'start', lcr_aer
    processes = []
    processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, new_left, english, french, iterations, test_data, model_name+str(new_left), lcr_aer, 1)))
    if current != new_left:
      processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, current, english, french, iterations, test_data, model_name+str(current), lcr_aer, 2)))
    if new_right != current:
      processes.append(Process(target = _climb_once, args = (model_type, e_vocab, f_vocab, new_right, english, french, iterations, test_data, model_name+str(new_right), lcr_aer, 3)))
    for p in processes:
      p.start()
    for p in processes:
      p.join()
    
    if current == new_left:
      lcr_aer[2] = lcr_aer[1]
    if new_right == current:
      lcr_aer[3] = lcr_aer[2]
    
    # Let cores cool off
    time.sleep(TIME)
    #lcr_aer = manager.list([0.3943, 0.56, 0.6179, 0.64, 0.67])
    
    #print new_left_aer, current_aer, new_right_aer
    #prepare for next iteration
    #lcr_aer = [new_left_aer, current_aer, new_right_aer]
    lcr = [left, new_left, current, new_right, right]
    print lcr
    print lcr_aer
    sys.stdout.flush()
    
    if all(aer == lcr_aer[0] for aer in lcr_aer):
      left = lcr[0]
      right = lcr[0]
      left_aer = lcr_aer[0]
      right_aer = lcr_aer[0]
      break

    min_aer = min(lcr_aer)
    mi = lcr_aer.index(min_aer)
    
    li = max(0, mi-1)
    ri = min(len(lcr_aer), mi+1)
    
    left = lcr[li]
    right = lcr[ri]
    current = int((left + right) / 2)
    new_left = int((left + current) / 2)
    new_right = int((current + right) / 2)
    
    left_aer = lcr_aer[li]
    right_aer = lcr_aer[ri]
    current_aer = 0
    if current in lcr:
      current_aer = lcr_aer[lcr.index(current)]
    new_left_aer = 0
    if new_left in lcr:
      new_left_aer = lcr_aer[lcr.index(new_left)]
    new_right_aer = 0
    if new_right in lcr:
      new_right_aer = lcr_aer[lcr.index(new_right)]
    lcr_aer = manager.list([left_aer, new_left_aer, current_aer, new_right_aer, right_aer])
    
    print left, current, right
    print left_aer, current_aer, right_aer
    sys.stdout.flush()

  if left_aer < right_aer:
    print left, left_aer
  else:
    print right, right_aer

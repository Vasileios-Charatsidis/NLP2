import sys
import codecs
from collections import defaultdict

import dataloader as dl

error = 'Usage: python check_nulls.py problem [train_corpus] corpus_file alignments_file true_alignments_file\n\
         problem could be either:\n\
         1. \'rare\' - then corpus_file should be the source corpus and train_corpus is expected\n\
         2. \'null\' - corpus_file should be the target corpus and train_corpus is not expected'

def count_word_appearances(train_corpus, vocab):
  word_counts = defaultdict(int)
  for sentence_no, sentence in enumerate(train_corpus):
    for word in sentence:
      word_counts[word] += 1
  word_counts_list = []
  wc = defaultdict(set)
  for word in word_counts:
    wc[word_counts[word]].add(word)
  return wc

def count_rare_word_alignments(corpus, rare, alignments, sure, prob):
  # corpus is source corpus (alignments are "t s [S|P])
  sentence_rare_counts = []
  for sentence_no, sentence in enumerate(corpus):
    s_sure = sure[sentence_no+1]
    s_prob = prob[sentence_no+1]
    s_als = alignments[sentence_no+1]
    rare_count = 0
    for al in s_als:
      word = sentence[al[1]-1]
      if word in rare and al not in s_sure and al not in s_prob:
        rare_count += 1
    sentence_rare_counts.append( (rare_count, sentence_no+1) )
  sentence_rare_counts.sort(reverse=True)
  return sentence_rare_counts

def is_aligned_to_null(eng, true_als):
  for al in true_als:
    # aligned to something non-null
    if eng == al[0]:
      return False
  return True

def count_nulls_in_true_alignments(corpus, alignments, sure, prob):
  # corpus is target corpus (alignments are "t s [S|P])
  null_all_sentence = []
  for sentence_no, sentence in enumerate(corpus):
    s_sure = sure[sentence_no+1]
    s_prob = prob[sentence_no+1]
    s_als = alignments[sentence_no+1]
    not_aligned_to_null_count = 0
    for al in s_als:
      e = al[0]
      al_to_null = is_aligned_to_null(e, s_sure)
      al_to_null = al_to_null and is_aligned_to_null(e, s_prob)
      if al_to_null:
        not_aligned_to_null_count += 1
    null_all_sentence.append( (not_aligned_to_null_count, sentence_no+1) )
  null_all_sentence.sort(reverse=True)
  return null_all_sentence

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print error
    sys.exit()

  problem = sys.argv[1].lower()
  if 'rare' != problem and 'null' != problem:
    print error
    sys.exit()

  if 'rare' == problem:
    if len(sys.argv) < 6:
      print error
      sys.exit()
    train_corpus_fname = sys.argv[2]
    corpus_fname = sys.argv[3]
    alignments_fname = sys.argv[4]
    true_alignments_fname = sys.argv[5]
    
    train_corpus, vocab = dl.read(train_corpus_fname)
    corpus, _ = dl.read(corpus_fname)
    alignments, _ = dl.read_sentence_alignments(alignments_fname)
    sure, prob = dl.read_sentence_alignments(true_alignments_fname)
    
    word_counts = count_word_appearances(train_corpus, vocab)
    wcs = word_counts.keys()
    wcs.sort()
    #accumulated = 0
    #for wc in wcs:
    #  accumulated += len(word_counts[wc])
    #  print wc, len(word_counts[wc]), len(vocab), 100 * accumulated / float(len(vocab))
    rare = set()
    for wc in wcs:
      if wc > 5:
        break
      rare = rare.union(word_counts[wc])
    print len(rare), 100 * len(rare) / float(len(vocab))
    counts = count_rare_word_alignments(corpus, rare, alignments, sure, prob)
    for count in counts:
      print 'Sentence', count[1], '\tincorrect alignments to rare words', count[0]
  else:
    if len(sys.argv) < 5:
      print error
      sys.exit()
    corpus_fname = sys.argv[2]
    alignments_fname = sys.argv[3]
    true_alignments_fname = sys.argv[4]
  
    corpus, vocab = dl.read(corpus_fname)
    alignments, _ = dl.read_sentence_alignments(alignments_fname)
    sure, prob = dl.read_sentence_alignments(true_alignments_fname)
    
    not_null_all_sentence = count_nulls_in_true_alignments(corpus, alignments, sure, prob)
    for null_all in not_null_all_sentence:
      print 'Sentence', null_all[1], 'words wrongly not aligned to null', null_all[0]


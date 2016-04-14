import sys
from collections import defaultdict

error = 'Usage: python rank_sentences.py ibm1_alignments extension_alignments true_alignments'

def read_alignments(alignment_file_name):
  alignment_file = open(alignment_file_name, 'r')
  sure_alignments = defaultdict(set)
  prob_alignments = defaultdict(set)
  for alignment in alignment_file:
    #0001 1 1 S
    tokens = alignment.strip('\n').split()
    sentence_no = int(tokens[0])
    e = int(tokens[1])
    f = int(tokens[2])
    if len(tokens) <= 3 or 's' == tokens[3].lower():
      sure_alignments[sentence_no].add( (e, f) )
    else:
      prob_alignments[sentence_no].add( (e, f) )
  alignment_file.close()
  return sure_alignments, prob_alignments

def compute_sentence_aer(alignments, sure, possible):
  probable = sure.union(possible)
  correct_sure = len(alignments.intersection(sure))
  correct_probable = len(alignments.intersection(probable))
  aer = 1 - float(correct_sure + correct_probable) / float(len(alignments) + len(sure))
  return aer

def compute_aers(alignments, sure, prob):
  aers = dict()
  for sentence_no in ibm1_alignments:
    aer = compute_sentence_aer(alignments[sentence_no], sure[sentence_no], prob[sentence_no])
    aers[sentence_no] = aer
  return aers

def compute_diffs(ibm1_aers, ext_aers):
  diffs = []
  for sentence_no in ibm1_aers:
    diff = ext_aers[sentence_no] - ibm1_aers[sentence_no]
    diffs.append( (diff, sentence_no) )
  return diffs

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print error
    sys.exit()

  ibm1_al_fname = sys.argv[1]
  ext_al_fname = sys.argv[2]
  true_al_fname = sys.argv[3]

  ibm1_alignments, _ = read_alignments(ibm1_al_fname)
  extension_alignments, _ = read_alignments(ext_al_fname)
  sure_alignments, prob_alignments = read_alignments(true_al_fname)

  ibm1_aers = compute_aers(ibm1_alignments, sure_alignments, prob_alignments)
  ext_aers = compute_aers(extension_alignments, sure_alignments, prob_alignments)

  aer_diffs = compute_diffs(ibm1_aers, ext_aers)

  # sort descending based on difference and return indices
  aer_diffs.sort(reverse=True)
  for diff in aer_diffs:
    print diff[0], diff[1]

import sys
import os
import gzip
import string
from multiprocessing import Process, Manager
from nltk.tag.stanford import StanfordPOSTagger

from translation import Translation

error = 'Usage: pretag.py sentences out_dir language from_sentence to_sentence process_no'


def pos_tag(sentences_fname, out_dir, tagger, from_sentence, step):
    sentences_file = gzip.open(sentences_fname, 'r')
    to_sentence = from_sentence + step
    out_fname_current = out_dir + '/' + str(from_sentence) + '_' + str(to_sentence)
    tagged_sentences_file = open(out_fname_current, 'w')

    for sentence in sentences_file:
        if len(sentence.strip(string.whitespace)) == 0:
            continue
        sentence = sentence.decode('utf-8')
        tokens = sentence.strip(string.whitespace).split('|||')
        sentence_no = int(tokens[0])
        if sentence_no < from_sentence:
            continue
        if sentence_no >= to_sentence:
            break
        translation = Translation(tokens, '', '')
        pos_tags = ' '.join(map(lambda x: x[1], tagger.tag(translation.translation)))
        sentence = str(sentence_no) + ' ||| ' + pos_tags
        tagged_sentences_file.write(sentence.encode('utf-8'))
        tagged_sentences_file.write('\n')
        tagged_sentences_file.flush()

    sentences_file.close()
    tagged_sentences_file.close()


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print error
        sys.exit()

    sentences_fname = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    st = None
    if sys.argv[3] == 'german':
        st = StanfordPOSTagger('stanford-postagger-full-2015-12-09/models/german-fast.tagger',
                               'stanford-postagger-full-2015-12-09/stanford-postagger.jar')
    else:
        st = StanfordPOSTagger('stanford-postagger-full-2015-12-09/models/wsj-0-18-bidirectional-distsim.tagger',
                               'stanford-postagger-full-2015-12-09/stanford-postagger.jar')
    from_sentence = int(sys.argv[4])
    to_sentence = int(sys.argv[5])
    process_no = int(sys.argv[6])

    step = 1
    current_sentences = []
    for i in range(from_sentence, to_sentence, step):
        current_sentences.append(i)
        if len(current_sentences) == process_no:
            processes = []
            for cs in current_sentences:
                processes.append(Process(target=pos_tag, args=(sentences_fname, out_dir, st, cs, step)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            current_sentences = list()

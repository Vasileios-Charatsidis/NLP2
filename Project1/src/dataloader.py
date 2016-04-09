import codecs

#1 underscore means you are not enouraged to use this outside of this module
def _read(file_name):
  f = codecs.open(file_name, mode='r', encoding='utf-8')
  vocab = set()
  corpus = list()
  for sentence in f:
    # TODO: Get word roots/lemmas. Should be better
    sentence_words = map(lambda x: x.lower(), sentence.strip(' \n').split(' '))
    corpus.append(sentence_words)
    for word in sentence_words:
      vocab.add(word)
  return corpus, vocab

#2 underscores mean you are really not supposed to use this outside of this module
def __print_corpus(corpus):
  for sentence in corpus:
    print sentence

def read_data(english_file_name, french_file_name):
  english, e_vocab = _read(english_file_name)
  french, f_vocab = _read(french_file_name)
  
#  __print_corpus(english)
#  __print_corpus(french)
  
  return english, french, e_vocab, f_vocab
  

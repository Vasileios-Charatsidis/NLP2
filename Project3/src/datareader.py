import string
import gzip

from translation import Translation


class Reader:

    def __init__(self, translations_fname, references_fname):
        self.translations_fname = translations_fname
        self.references_fname = references_fname
        self.translations_file = None
        self.references_file = None
        self.last_translation = None
        self.restart()

    def read_next_src_nbest_translations(self):
        reference = self.references_file.readline()
        if not reference or reference == '\n':
            print self.last_translation[0]
            self.close()
            return None
        reference = reference.decode('utf-8').strip()

        translations = list()
        cur_sentence_no = 0
        if self.last_translation:
            translations.append(Translation(self.last_translation, reference))
            cur_sentence_no = int(self.last_translation[0])
        for nbest_translation in self.translations_file:
            nbest_translation = nbest_translation.decode('utf-8')
            tokens = nbest_translation.strip(string.whitespace).split('|||')
            sentence_no = int(tokens[0])
            assert(sentence_no >= cur_sentence_no)
            if sentence_no > cur_sentence_no:
                self.last_translation = tokens
                break
            translations.append(Translation(tokens, reference))
        return translations

    def skip_next_src_nbest_translations(self):
        reference = self.references_file.readline()
        if not reference or reference == '\n':
            self.close()
            return False

        cur_sentence_no = 0
        if self.last_translation:
            cur_sentence_no = int(self.last_translation[0])
            print cur_sentence_no
        for nbest_translation in self.translations_file:
            nbest_translation = nbest_translation.decode('utf-8')
            tokens = nbest_translation.strip(string.whitespace).split('|||')
            sentence_no = int(tokens[0])
            assert(sentence_no >= cur_sentence_no)
            if sentence_no > cur_sentence_no:
                self.last_translation = tokens
                break
        return True

    def restart(self):
        self.translations_file = gzip.open(self.translations_fname, 'r')
        self.references_file = gzip.open(self.references_fname, 'r')
        self.last_translation = None

    def close(self):
        self.translations_file.close()
        self.references_file.close()
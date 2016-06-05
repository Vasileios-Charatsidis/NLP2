import string
import gzip

from translation import Translation


class Reader:

    def __init__(self, translations_fname, references_fname, pos_tags_fname):
        self.translations_fname = translations_fname
        self.references_fname = references_fname
        self.pos_tags_fname = pos_tags_fname
        self.translations_file = None
        self.references_file = None
        self.pos_tags_file = None
        self.last_translation = None
        self.last_pos_tag = None
        self.pos_tags_eof = False
        self.restart()

    def _read_reference(self):
        reference = self.references_file.readline()
        return None if reference == '\n' else reference

    def _read_pos_tags(self):
        pos_tags = list()
        if self.pos_tags_eof:
            return pos_tags
        cur_sentence_no = 0
        if self.last_translation:
            pos_tags.append(self.last_pos_tag[1].strip(string.whitespace))
            cur_sentence_no = int(self.last_pos_tag[0])
        self.pos_tags_eof = True
        for pos_tag in self.pos_tags_file:
            pos_tag = pos_tag.decode('utf-8')
            tokens = pos_tag.strip(string.whitespace).split('|||')
            sentence_no = int(tokens[0])
            assert (sentence_no >= cur_sentence_no)
            if sentence_no > cur_sentence_no:
                self.last_pos_tag = tokens
                self.pos_tags_eof = False
                break
            pos_tags.append(tokens[1].strip(string.whitespace))
        if self.pos_tags_eof:
            self.last_pos_tag = None
        return pos_tags

    def _read_translations(self, reference, pos_tags):
        translations = list()
        cur_sentence_no = 0
        translation_count = 0
        if self.last_translation:
            translations.append(Translation(self.last_translation, reference, pos_tags[translation_count]))
            cur_sentence_no = int(self.last_translation[0])
            translation_count += 1
        for nbest_translation in self.translations_file:
            nbest_translation = nbest_translation.decode('utf-8')
            tokens = nbest_translation.strip(string.whitespace).split('|||')
            sentence_no = int(tokens[0])
            assert (sentence_no >= cur_sentence_no)
            if sentence_no > cur_sentence_no:
                self.last_translation = tokens
                break
            translations.append(Translation(tokens, reference, pos_tags[translation_count]))
            translation_count += 1
        return translations

    def read_next_src_nbest_translations(self):
        reference = self._read_reference()
        if not reference:
            self.close()
            return None

        pos_tags = self._read_pos_tags()
        if not pos_tags:
            self.close()
            return None

        translations = self._read_translations(reference, pos_tags)
        return translations

    def skip_next_src_nbest_translations(self):
        reference = self._read_reference()
        if not reference:
            self.close()
            return False

        pos_tags = self._read_pos_tags()
        if not pos_tags:
            self.close()
            return False

        cur_sentence_no = 0
        if self.last_translation:
            cur_sentence_no = int(self.last_translation[0])
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
        self.pos_tags_file = open(self.pos_tags_fname, 'r')
        self.last_translation = None
        self.last_pos_tag = None
        self.pos_tags_eof = False

    def close(self):
        self.translations_file.close()
        self.references_file.close()
        self.pos_tags_file.close()
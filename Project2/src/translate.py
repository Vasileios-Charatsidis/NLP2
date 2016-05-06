import sys

import common
import inputreader as ir
import translationprocessor as tp


error = 'Usage: python translate.py source_sentences phrase_tables_dir sentence_count weights_file derivations_count\
 translation_type [permutations_file] output_dir\n\
 \t- translation_type could be either \'monotone\' or \'lattice\'.\
 In case of \'lattice\', \'permutations_file\' is expected'


def parse_arguments():
    tr_type = sys.argv[6]
    perms_fname = None
    if 'monotone' == tr_type:
        out_dir = sys.argv[7]
    elif len(sys.argv) < 9 or 'lattice' != tr_type:
        print error
        sys.exit()
    else:
        perms_fname = sys.argv[7]
        out_dir = sys.argv[8]
    return sys.argv[1], sys.argv[2], int(sys.argv[3]),\
        sys.argv[4], int(sys.argv[5]), tr_type, perms_fname, out_dir


if __name__ == '__main__':
    if len(sys.argv) < 8:
        print error
        sys.exit()

    source_sentences_fname, phrase_tables_dir, sentences_count, weights_fname, derivations_count,\
        translation_type, permutations_fname, output_dir = parse_arguments()

    source_sentences, phrase_table_fnames, weights, permutations_per_sentence =\
        ir.read_input(source_sentences_fname, phrase_tables_dir, sentences_count, weights_fname, permutations_fname)

    # Create output folder
    common.makedir(output_dir)

    tp.translate(source_sentences, phrase_table_fnames, weights, translation_type,
                 permutations_per_sentence, derivations_count, output_dir)

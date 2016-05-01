import sys
import os
import string
import common


error = 'Usage: python encode_source_as_transducers.py input_sentences output_folder sentence_count'


def create_fst_binary(sentence, sentence_no, output_dir):
    # Create file names
    fst_txt_name = common.make_path_name(output_dir, 'fst_txt', sentence_no)
    fst_bin_name = common.make_path_name(output_dir, 'fst_bin', sentence_no)
    isymb_txt_name = common.make_path_name(output_dir, 'isymb', sentence_no)
    osymb_txt_name = common.make_path_name(output_dir, 'osymb', sentence_no)
    # Open files
    fst_txt = common.open_utf(fst_txt_name, 'w')
    isymb_txt = common.open_utf(isymb_txt_name, 'w')
    osymb_txt = common.open_utf(osymb_txt_name, 'w')
    # Write to files
    # Add epsilon to symbol files just in case
    isymb_txt.write(common.SYMB_TEMPLATE.format(common.EPSILON, 0))
    osymb_txt.write(common.SYMB_TEMPLATE.format(common.EPSILON, 0))
    word_ids = dict()
    for pos, word in enumerate(sentence):
        fst_txt.write(common.FST_TEMPLATE.format(pos, pos+1, pos+1, word))
        isymb_txt.write(common.SYMB_TEMPLATE.format(str(pos+1), pos+1))
        word_ids[word] = pos + 1
    for word in word_ids:
        osymb_txt.write(common.SYMB_TEMPLATE.format(word, word_ids[word]))
    fst_txt.write(str(len(sentence)) + '\n')
    fst_txt.close()
    isymb_txt.close()
    osymb_txt.close()
    # Compile text files in binary fst
    common.make_fst(fst_txt_name, fst_bin_name, isymb_txt_name, osymb_txt_name)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print error
        sys.exit()

    source = sys.argv[1]
    output_dir = sys.argv[2]
    sentence_count = int(sys.argv[3])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    source_file = common.open_utf(source, 'r')

    sentence_no = 0
    for sentence_line in source_file:
        if sentence_no >= sentence_count:
            break
        sentence = sentence_line.strip(string.whitespace).split()
        create_fst_binary(sentence, sentence_no, output_dir)
        sentence_no += 1

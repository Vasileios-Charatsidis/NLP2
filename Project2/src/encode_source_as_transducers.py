import sys
import os
import subprocess
import codecs
import string

error = 'Usage: python encode_source_as_transducers.py input_sentences output_folder sentence_count'

EPSILON = '<eps>'
FST_TEMPLATE = '{0:d} {1:d} {1:d} {2}\n'
WORD_SYMB_TEMPLATE = '{0} {1:d}\n'
NUM_SYMB_TEMPLATE = '{0:d} {0:d}\n'

def make_path_name(directory, name, extension):
    return '{0}/{1:d}{2}'.format(directory, name, extension)


def open_file(fname, mode):
    return codecs.open(fname, mode, encoding='utf8')


def construct_callee(fst_txt, fst_bin, isymb, osymb):
    # fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
    callee = ['fstcompile']
    callee.append('--isymbols=' + isymb)
    callee.append('--osymbols=' + osymb)
    callee.append('--keep_isymbols')
    callee.append('--keep_osymbols')
    callee.append(fst_txt)
    callee.append(fst_bin)
    return callee


def create_fst_binary(sentence, sentence_no, output_dir):
    # Create file names
    fst_txt_name = make_path_name(output_dir, sentence_no, 'fst.txt')
    fst_bin_name = make_path_name(output_dir, sentence_no, 'fst.fst')
    isymb_txt_name = make_path_name(output_dir, sentence_no, 'isymb.txt')
    osymb_txt_name = make_path_name(output_dir, sentence_no, 'osymb.txt')
    # Open files
    fst_txt = open_file(fst_txt_name, 'w')
    isymb_txt = open_file(isymb_txt_name, 'w')
    osymb_txt = open_file(osymb_txt_name, 'w')
    # Write to files
    # Add epsilon to symbol files just in case
    isymb_txt.write(WORD_SYMB_TEMPLATE.format(EPSILON, 0))
    osymb_txt.write(WORD_SYMB_TEMPLATE.format(EPSILON, 0))
    for pos, word in enumerate(sentence):
        fst_txt.write(FST_TEMPLATE.format(pos, pos+1, word))
        isymb_txt.write(NUM_SYMB_TEMPLATE.format(pos+1))
        osymb_txt.write(WORD_SYMB_TEMPLATE.format(word, pos+1))
    fst_txt.write(str(len(sentence)) + '\n')
    fst_txt.close()
    isymb_txt.close()
    osymb_txt.close()
    # Compile text files in binary fst
    callee = construct_callee(fst_txt_name, fst_bin_name, isymb_txt_name, osymb_txt_name)
    subprocess.call(callee)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print error
        sys.exit()

    source = sys.argv[1]
    output_dir = sys.argv[2]
    sentence_count = int(sys.argv[3])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    source_file = open_file(source, 'r')

    sentence_no = 0
    for sentence_line in source_file:
        if sentence_no >= sentence_count:
            break
        sentence = sentence_line.strip(string.whitespace).split()
        create_fst_binary(sentence, sentence_no, output_dir)
        sentence_no += 1

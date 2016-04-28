import codecs
import os
import subprocess


EPSILON = u'<eps>'
FST_TEMPLATE = u'{0:d} {1:d} {1:d} {2}\n'
FST_WEIGHTED_TEMPLATE = u'{0:d} {1:d} {2} {3} {4:f}\n'
WORD_SYMB_TEMPLATE = u'{0} {1:d}\n'
NUM_SYMB_TEMPLATE = u'{0:d} {0:d}\n'


def make_path_name(directory, name, extension):
    return '{0}/{1}.{2:d}'.format(directory, name, extension)


def open_utf(fname, mode):
    return codecs.open(fname, mode, encoding='utf8')


def list_filter_sort_filenames(folder, filter):
    # Get all files from the dir
    filenames_all = [os.path.join(folder, f) for f in os.listdir(folder)]
    # Filter out the ones that are not ordinary files
    filenames = [f for f in filenames_all if filter(f)]
    # Sort them based on corresponding sentence number (which is extension)
    filenames.sort(key=lambda f: int(f[f.rfind('.') + 1:]))
    return filenames


def make_fst(fst_txt, fst_bin, isymb, osymb):
    # fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
    callee = ['fstcompile']
    callee.append('--isymbols=' + isymb)
    callee.append('--osymbols=' + osymb)
    callee.append('--keep_isymbols')
    callee.append('--keep_osymbols')
    callee.append(fst_txt)
    callee.append(fst_bin)
    subprocess.call(callee)


def compose_fsts(first, second, output):
    # fstcompose first second output
    callee = ['fstcompose', first, second, output]
    subprocess.call(callee)

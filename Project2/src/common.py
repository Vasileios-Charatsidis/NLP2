import codecs
import os
import subprocess
import string


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


def get_best_fst(output_fst_fname, best_fst_fname, derivations_count):
    # fstshortestpath --nshortest=100 fst > best_fst
    callee = ['fstshortestpath', '--nshortest={0:d}'.format(derivations_count), output_fst_fname]
    best_fst_file = open_utf(best_fst_fname, 'w')
    subprocess.call(callee, stdout=best_fst_file)
    best_fst_file.close()


def get_best_derivations(best_fst_fname, best_derivations_fname):
    # fstprint fst > text_fst
    callee = ['fstprint', best_fst_fname]
    best_derivations_file = open_utf(best_derivations_fname, 'w')
    subprocess.call(callee, stdout=best_derivations_file)
    best_derivations_file.close()


def get_best_derivations_h(best_derivations_fname, best_derivations_h_fname):
    best_derivations_file = open_utf(best_derivations_fname, 'r')
    start_transitions, transitions, finals = read_derivations(best_derivations_file)
    best_derivations_file.close()
    best_derivations_h_file = open_utf(best_derivations_h_fname, 'w')
    # len(start_transitions) > 0
    while start_transitions:
        start_transition = start_transitions.pop()
        aligned_translation = get_aligned_translation(start_transition, transitions, finals)
        best_derivations_h_file.write(aligned_translation + '\n')
    best_derivations_h_file.close()


def read_derivations(best_derivations_file):
    # Do not convert anything to int
    start_node = None
    start_transitions = set()
    finals = set()
    transitions = dict()
    for line in best_derivations_file:
        line = line.strip(string.whitespace).split('\t')
        # Escape empty lines if any
        if len(line) < 1:
            continue
        # collect final states (usually it's just one and it's '1')
        elif 1 == len(line):
            finals.add(line[0])
        else:
            if start_node is None:
                start_node = line[0]
            # Collect transitions that start a sentence in a different set
            if line[0] == start_node:
                start_transitions.add((line[1], line[2], line[3]))
            # from key go to 1st el of tuple, second and third elements are alignment and tgt word
            else:
                transitions[line[0]] = (line[1], line[2], line[3])
    return start_transitions, transitions, finals


def get_aligned_translation(start_transition, transitions, finals):
    assert(start_transition[1] == EPSILON)
    assert(start_transition[2] == EPSILON)
    next_state = start_transition[0]
    aligned_sentence = ''
    while next_state not in finals:
        aligned_phrase, next_state = get_next_phrase(next_state, transitions)
        aligned_sentence += aligned_phrase
        if next_state not in finals:
            aligned_sentence += ' '
    return aligned_sentence


def get_next_phrase(next_state, transitions):
    transition = transitions[next_state]
    aligned_phrase = ''
    # in case of other empty transitions
    if transition[1] == EPSILON:
        assert(transition[2] == EPSILON)
        return '', transition[0]
    # singleton phrase
    if transition[2] != EPSILON:
        aligned_phrase += u'{0} |{1}-{1}|'.format(transition[2], transition[1])
        return aligned_phrase, transition[0]
    # phrases with more than 1 words
    align_start = transition[1]
    align_end = align_start
    while transition[1] != EPSILON:
        align_end = transition[1]
        next_state = transition[0]
        transition = transitions[next_state]
    while transition[1] == EPSILON:
        if transition[2] != EPSILON:
            aligned_phrase += transition[2] + ' '
        next_state = transition[0]
        # Final state is not a key in transitions
        if next_state in transitions:
            transition = transitions[next_state]
        else:
            break
    aligned_phrase += '|{0}-{1}|'.format(align_start, align_end)
    return aligned_phrase, next_state

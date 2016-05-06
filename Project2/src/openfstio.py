import math
import string
import subprocess
import os

import common


EPSILON = u'<eps>'
FST_TEMPLATE = u'{0:d} {1:d} {2} {3}\n'
FST_WEIGHTED_TEMPLATE = u'{0:d} {1:d} {2} {3} {4:f}\n'
SYMB_TEMPLATE = u'{0} {1:d}\n'


def write_symbol_file(symbols_fname, vocab, map_to_self):
    symbol_file = common.open_utf(symbols_fname, 'w')
    symbol_mapping = common.make_symbol_mapping(vocab, map_to_self)
    # Add epsilon to symbol files just in case
    symbol_file.write(SYMB_TEMPLATE.format(EPSILON, 0))
    for symbol in symbol_mapping:
        symbol_file.write(SYMB_TEMPLATE.format(symbol, symbol_mapping[symbol]))
    symbol_file.close()


def write_sentence(fst_txt_fname, sentence):
    fst_txt_file = common.open_utf(fst_txt_fname, 'w')
    for src, tgt in enumerate(sentence):
        fst_txt_file.write(FST_TEMPLATE.format(src, src + 1, src, tgt))
    fst_txt_file.write(str(len(sentence)) + '\n')
    fst_txt_file.close()


def write_permutation(fst_txt_file, permutation, permuted_words, prob, weight, start_state, next_state):
    assert (len(permutation) == len(permuted_words))
    first = True
    for index, word in zip(permutation, permuted_words)[:-1]:
        if first:
            fst_txt_file.write(FST_TEMPLATE.format(start_state, next_state, index, word))
            first = False
        else:
            fst_txt_file.write(FST_TEMPLATE.format(next_state - 1, next_state, index, word))
        next_state += 1
    index = permutation[-1]
    word = permuted_words[-1]
    fst_txt_file.write(
        FST_WEIGHTED_TEMPLATE.format(next_state - 1, next_state, str(index), word, -math.log(prob) * weight))
    fst_txt_file.write(str(next_state) + '\n')
    return next_state + 1


def write_phrase(fst_file, src_phrase, tgt_phrase, features, weights, state_id):
    weight = common.calculate_weight(features, weights)
    if 1 == len(src_phrase) and 1 == len(tgt_phrase):
        fst_file.write(FST_WEIGHTED_TEMPLATE.format(0, 0, src_phrase[0], tgt_phrase[0], weight))
    else:
        for i, src_word in enumerate(src_phrase):
            if 0 == i:
                fst_file.write(FST_WEIGHTED_TEMPLATE.format(0, state_id, src_word, EPSILON, weight))
            else:
                weight = 0
                fst_file.write(
                    FST_WEIGHTED_TEMPLATE.format(state_id, state_id + 1, src_word, EPSILON, weight))
                state_id += 1
        weight = 0
        for tgt_word in tgt_phrase[:-1]:
            fst_file.write(FST_WEIGHTED_TEMPLATE.format(state_id, state_id + 1, EPSILON, tgt_word, weight))
            state_id += 1
        fst_file.write(FST_WEIGHTED_TEMPLATE.format(state_id, 0, EPSILON, tgt_phrase[-1], weight))
        state_id += 1
    return state_id


def write_pt_fst_file(fst_fname, phrases, unknown_words, weights):
    fst_file = common.open_utf(fst_fname, 'w')
    state_id = 1
    for phrase in phrases:
        state_id = write_phrase(fst_file, phrase[0], phrase[1], phrase[2], weights, state_id)
    for word in unknown_words:
        write_phrase(fst_file, [word], [word], {'PassThrough': 1}, weights, 0)
    # Mark starting state as final
    fst_file.write('0\n')
    fst_file.close()


def make_fst(fst_txt, fst_bin, isymb, osymb):
    # fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
    callee = ['fstcompile', '--isymbols=' + isymb, '--osymbols=' + osymb,
              '--keep_isymbols', '--keep_osymbols', fst_txt, fst_bin]
    subprocess.call(callee)


def sort_fst_arcs(fst_bin_name, sort_type):
    fst_temp_fname = fst_bin_name + '.temp'
    subprocess.call(['fstarcsort', '--sort_type='+sort_type, fst_bin_name, fst_temp_fname])
    os.remove(fst_bin_name)
    os.rename(fst_temp_fname, fst_bin_name)


def compose_fsts(first, second, output):
    # fstcompose first second output
    callee = ['fstcompose', first, second, output]
    subprocess.call(callee)


def get_best_fst(output_fst_fname, best_fst_fname, derivations_count):
    # fstshortestpath --nshortest=100 fst > best_fst
    callee = ['fstshortestpath', '--nshortest={0:d}'.format(derivations_count), output_fst_fname]
    best_fst_file = common.open_utf(best_fst_fname, 'w')
    subprocess.call(callee, stdout=best_fst_file)
    best_fst_file.close()


def determinize_and_minimize(fst_bin_fname):
    # fstrmepsilon in | fstdeterminize | fstpush --push_weights=true | fstminimize | fsttopsort out
    fst_no_eps_fname = fst_bin_fname.replace('bin', 'no_eps')
    subprocess.call(['fstrmepsilon', fst_bin_fname, fst_no_eps_fname])
    fst_det_fname = fst_bin_fname.replace('bin', 'det')
    subprocess.call(['fstdeterminize', fst_no_eps_fname, fst_det_fname])
    fst_pushed_fname = fst_bin_fname.replace('bin', 'pushed')
    subprocess.call(['fstpush', '--push_weights=true', fst_det_fname, fst_pushed_fname])
    fst_min_fname = fst_bin_fname.replace('bin', 'min')
    subprocess.call(['fstminimize', fst_pushed_fname, fst_min_fname])
    os.remove(fst_bin_fname)
    subprocess.call(['fsttopsort', fst_min_fname, fst_bin_fname])
    os.remove(fst_no_eps_fname)
    os.remove(fst_pushed_fname)
    os.remove(fst_det_fname)
    os.remove(fst_min_fname)


def get_best_derivations(best_fst_fname, best_derivations_fname):
    # fstprint fst > text_fst
    callee = ['fstprint', best_fst_fname]
    best_derivations_file = common.open_utf(best_derivations_fname, 'w')
    subprocess.call(callee, stdout=best_derivations_file)
    best_derivations_file.close()


# Expects get_best_derivations to have been called already
def get_best_derivations_h(best_derivations_fname, best_derivations_h_fname):
    best_derivations_file = common.open_utf(best_derivations_fname, 'r')
    start_transitions, transitions, finals = read_derivations(best_derivations_file)
    best_derivations_file.close()
    best_derivations_h_file = common.open_utf(best_derivations_h_fname, 'w')
    # len(start_transitions) > 0
    while start_transitions:
        start_transition = start_transitions.pop()
        aligned_translation, _ = get_translation(start_transition, transitions, finals, True)
        best_derivations_h_file.write(aligned_translation + '\n')
    best_derivations_h_file.close()


def get_translation(start_transition, transitions, finals, aligned=False):
    assert(start_transition[1] == EPSILON)
    assert(start_transition[2] == EPSILON)
    next_state = start_transition[0]
    sentence = ''
    sentence_cost = 0
    while next_state not in finals:
        phrase, next_state, cost = get_next_phrase(next_state, transitions, aligned)
        sentence += phrase
        sentence_cost += cost
        if next_state not in finals and phrase != '':
            sentence += ' '
    return sentence, sentence_cost


def get_next_phrase(next_state, transitions, aligned):
    transition = transitions[next_state]
    phrase = ''
    cost = 0
    # in case of other empty transitions
    if transition[1] == EPSILON:
        assert(transition[2] == EPSILON)
        assert(3 == len(transition))
        return phrase, transition[0], cost
    # singleton phrase
    if transition[2] != EPSILON:
        if aligned:
            phrase += u'{0} |{1}-{1}|'.format(transition[2], transition[1])
        else:
            phrase += transition[2]
        assert(4 == len(transition))
        return phrase, transition[0], float(transition[3])
    # phrases with more than 1 words
    align_start = transition[1]
    align_end = align_start
    while transition[1] != EPSILON:
        align_end = transition[1]
        next_state = transition[0]
        # It will only be the first one, but too much trouble to assert that.
        if len(transition) > 3:
            cost += float(transition[3])
        transition = transitions[next_state]
    while transition[1] == EPSILON:
        if transition[2] != EPSILON:
            phrase += transition[2] + ' '
        next_state = transition[0]
        assert (3 == len(transition))
        # Final state is not a key in transitions
        if next_state in transitions:
            transition = transitions[next_state]
        else:
            break
    if aligned:
        phrase += '|{0}-{1}|'.format(align_start, align_end)
    else:
        assert(' ' == phrase[-1])
        phrase = phrase[:-1]
    return phrase, next_state, cost


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
                assert(4 == len(line))
                start_transitions.add((line[1], line[2], line[3]))
            # from key go to 1st el of tuple, second and third elements are alignment and tgt word
            else:
                if 4 == len(line):
                    transitions[line[0]] = (line[1], line[2], line[3])
                else:
                    transitions[line[0]] = (line[1], line[2], line[3], line[4])
    return start_transitions, transitions, finals

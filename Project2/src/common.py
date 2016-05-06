import codecs
import os
import subprocess
import string


EPSILON = u'<eps>'
FST_TEMPLATE = u'{0:d} {1:d} {2} {3}\n'
FST_WEIGHTED_TEMPLATE = u'{0:d} {1:d} {2} {3} {4:f}\n'
SYMB_TEMPLATE = u'{0} {1:d}\n'


def make_path_name(directory, name, extension):
    return '{0}/{1}.{2:d}'.format(directory, name, extension)


def open_utf(fname, mode):
    return codecs.open(fname, mode, encoding='utf8')


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def list_filter_filenames(folder, filter_func):
    # Get all files from the dir
    filenames_all = [os.path.join(folder, f) for f in os.listdir(folder)]
    # Filter out the ones that are not ordinary files
    filenames = [f for f in filenames_all if filter_func(f)]
    # Sort them based on corresponding sentence number (which is extension)
    filenames_dict = dict()
    for filename in filenames:
        filenames_dict[int(filename[filename.rfind('.') + 1:])] = filename
    return filenames_dict


def list_filter_sort_filenames(folder, filter_func):
    # Get all files from the dir
    filenames_all = [os.path.join(folder, f) for f in os.listdir(folder)]
    # Filter out the ones that are not ordinary files
    filenames = [f for f in filenames_all if filter_func(f)]
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
    best_fst_file = open_utf(best_fst_fname, 'w')
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
        aligned_translation, _ = get_translation(start_transition, transitions, finals, True)
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
                assert(4 == len(line))
                start_transitions.add((line[1], line[2], line[3]))
            # from key go to 1st el of tuple, second and third elements are alignment and tgt word
            else:
                if 4 == len(line):
                    transitions[line[0]] = (line[1], line[2], line[3])
                else:
                    transitions[line[0]] = (line[1], line[2], line[3], line[4])
    return start_transitions, transitions, finals


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


def get_best_translation_with_best_derivation(best_derivations_fname):
    best_derivations_file = open_utf(best_derivations_fname, 'r')
    start_transitions, transitions, finals = read_derivations(best_derivations_file)
    best_derivations_file.close()
    translations = dict()
    # len(start_transitions) > 0
    while start_transitions:
        start_transition = start_transitions.pop()
        translation, cost = get_translation(start_transition, transitions, finals)
        derivation, _ = get_translation(start_transition, transitions, finals, True)
        if translation not in translations:
            translations[translation] = list()
        translations[translation].append((cost, derivation))
    best_translation = None
    best_derivation = None
    best_cost = None
    for translation in translations:
        derivations = translations[translation]
        cost = reduce(lambda d1, d2: d1[0] + d2[0], derivations)
        if best_cost is None or best_cost > cost:
            best_cost = cost
            best_translation = translation
            best_tr_derivation = None
            best_der_cost = None
            for derivation in derivations:
                if best_der_cost is None or best_der_cost > derivation[0]:
                    best_der_cost = derivation[0]
                    best_tr_derivation = derivation[1]
            best_derivation = best_tr_derivation
    return best_translation, best_derivation

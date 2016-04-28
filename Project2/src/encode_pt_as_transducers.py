import sys
import os
import subprocess
import codecs
import string
import math

error = 'Usage: python encode_pt_as_transducers.py src_sentences phrase_tables_folder weights_file output_folder phrase_tables_count'

EPSILON = u'<eps>'
FST_TEMPLATE = u'{0:d} {1:d} {2} {3} {4:f}\n'
WORD_SYMB_TEMPLATE = u'{0} {1:d}\n'

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


def calculate_weight(tgt_words, features, weights):
    weight = 0
    for feature in features:
        weight += weights[feature] * features[feature]
    return weight


def process_phrase(fst_file, src_phrase, tgt_phrase, features, weights, state_id):
    weight = calculate_weight(tgt_phrase, features, weights)
    if 1 == len(src_phrase) and 1 == len(tgt_phrase):
        fst_file.write(FST_TEMPLATE.format(0, 0, src_phrase[0], tgt_phrase[0], weight))
    else:
        for i, src_word in enumerate(src_phrase):
            if 0 == i:
                fst_file.write(FST_TEMPLATE.format(0, state_id, src_word, EPSILON, weight))
            else:
                weight = 0
                fst_file.write(FST_TEMPLATE.format(state_id, state_id + 1, src_word, EPSILON, weight))
                state_id += 1
        weight = 0
        for tgt_word in tgt_phrase[:-1]:
            fst_file.write(FST_TEMPLATE.format(state_id, state_id + 1, EPSILON, tgt_word, weight))
            state_id += 1
        fst_file.write(FST_TEMPLATE.format(state_id, 0, EPSILON, tgt_phrase[-1], 0))
        state_id += 1
    return state_id


def write_fst_file(phrases, unknown_words, weights, fst_fname):
    fst_file = open_file(fst_fname, 'w')
    state_id = 1
    for phrase in phrases:
        state_id = process_phrase(fst_file, phrase[0], phrase[1], phrase[2], weights, state_id)
    for word in unknown_words:
        process_phrase(fst_file, [word], [word], {'PassThrough': 1}, weights, 0)
    # Mark starting state as final
    fst_file.write('0\n')
    fst_file.close()


def write_symbol_file(word_ids, symb_fname):
    symb_file = open_file(symb_fname, 'w')
    for word in word_ids:
        symb_file.write(WORD_SYMB_TEMPLATE.format(word, word_ids[word]))
    symb_file.close()


def create_word_ids(words):
    word_ids = dict()
    # Add epsilon
    word_id = 0
    word_ids[EPSILON] = word_id
    word_id += 1
    for word in words:
        word_ids[word] = word_id
        word_id += 1
    return word_ids


def collect_features(features_str, tgt_words):
    features = features_str.strip(string.whitespace).split()
    feature_dict = dict()
    for feature in features:
        feat = feature.split('=')
        feature_dict[feat[0]] = float(feat[1])
    feature_dict['Glue'] = 1
    feature_dict['WordPenalty'] = - len(tgt_words) / math.log(10)
    feature_dict['PassThrough'] = 0
    return feature_dict


def read_phrase_table(sentence_words, pt_fname):
    phrases = []
    known_src_words = set()
    known_tgt_words = set()
    pt_file = open_file(pt_fname, 'r')
    for line in pt_file:
        phrase = line.strip(string.whitespace).split(' ||| ')
        src_words = phrase[1].strip(string.whitespace).split()
        tgt_words = phrase[2].strip(string.whitespace).split()
        features = collect_features(phrase[3], tgt_words)
        known_src_words.update(src_words)
        known_tgt_words.update(tgt_words)
        phrases.append( (src_words, tgt_words, features) )
    pt_file.close()

    unknown_words = sentence_words.difference(known_src_words)
    src_word_ids = create_word_ids(sentence_words)
    known_tgt_words.update(unknown_words)
    tgt_word_ids = create_word_ids(known_tgt_words)
    return phrases, unknown_words, src_word_ids, tgt_word_ids


def process_setence(sentence_str, pt_fname, weights, sentence_no, output_dir):
    # Make a set of all words in the source sentence
    sentence_words = set(sentence_str.strip(string.whitespace).split())
    # Read the phrase table file
    phrases, unknown_words, src_word_ids, tgt_word_ids = read_phrase_table(sentence_words, pt_fname)
    # Create file names
    fst_bin_name = make_path_name(output_dir, sentence_no, 'fst.fst')
    fst_fname = make_path_name(output_dir, sentence_no, 'fst.txt')
    isymb_fname = make_path_name(output_dir, sentence_no, 'isymb.txt')
    osymb_fname = make_path_name(output_dir, sentence_no, 'osymb.txt')
    # Write to files
    write_symbol_file(src_word_ids, isymb_fname)
    write_symbol_file(tgt_word_ids, osymb_fname)
    write_fst_file(phrases, unknown_words, weights, fst_fname)
    # Compile text files in binary fst
    callee = construct_callee(fst_fname, fst_bin_name, isymb_fname, osymb_fname)
    subprocess.call(callee)


def read_weights(weights_fname):
    weights = dict()
    weights_file = open_file(weights_fname, 'r')
    for line in weights_file:
        line = line.strip(string.whitespace).split()
        if len(line) < 2:
            continue
        weights[line[0]] = float(line[1])
    weights_file.close()
    return  weights

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print error
        sys.exit()

    sentences_fname = sys.argv[1]
    pts_dir = sys.argv[2]
    weights_fname = sys.argv[3]
    output_dir = sys.argv[4]
    sentence_count = int(sys.argv[5])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read weights
    weights = read_weights(weights_fname)

    # Get all files from the input dir
    pts_all = [os.path.join(pts_dir, f) for f in os.listdir(pts_dir)]
    # Filter out the ones that are not ordinary files
    pts = [f for f in pts_all if os.path.isfile(f)]
    # Sort them based on corresponding sentence number
    pts.sort(key = lambda f: int(f[f.rfind('.')+1:]))

    # Open sentences and get ready to read
    sentences = open_file(sentences_fname, 'r')
    sentence_no = 0
    for pt in pts:
        if sentence_no >= sentence_count:
            break
        sentence = sentences.readline()
        process_setence(sentence, pt, weights, sentence_no, output_dir)
        sentence_no += 1

    sentences.close()

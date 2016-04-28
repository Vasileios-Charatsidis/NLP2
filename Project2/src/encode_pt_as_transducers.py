import sys
import os
import string
import math
import common

error = 'Usage: python encode_pt_as_transducers.py src_sentences phrase_tables_folder weights_file output_folder phrase_tables_count'


def calculate_weight(features, weights):
    weight = 0
    for feature in features:
        weight += weights[feature] * features[feature]
    return weight


def process_phrase(fst_file, src_phrase, tgt_phrase, features, weights, state_id):
    weight = calculate_weight(features, weights)
    if 1 == len(src_phrase) and 1 == len(tgt_phrase):
        fst_file.write(common.FST_WEIGHTED_TEMPLATE.format(0, 0, src_phrase[0], tgt_phrase[0], weight))
    else:
        for i, src_word in enumerate(src_phrase):
            if 0 == i:
                fst_file.write(common.FST_WEIGHTED_TEMPLATE.format(0, state_id, src_word, common.EPSILON, weight))
            else:
                weight = 0
                fst_file.write(common.FST_WEIGHTED_TEMPLATE.format(state_id, state_id + 1, src_word, common.EPSILON, weight))
                state_id += 1
        weight = 0
        for tgt_word in tgt_phrase[:-1]:
            fst_file.write(common.FST_WEIGHTED_TEMPLATE.format(state_id, state_id + 1, common.EPSILON, tgt_word, weight))
            state_id += 1
        fst_file.write(common.FST_WEIGHTED_TEMPLATE.format(state_id, 0, common.EPSILON, tgt_phrase[-1], 0))
        state_id += 1
    return state_id


def write_fst_file(phrases, unknown_words, weights, fst_fname):
    fst_file = common.open_utf(fst_fname, 'w')
    state_id = 1
    for phrase in phrases:
        state_id = process_phrase(fst_file, phrase[0], phrase[1], phrase[2], weights, state_id)
    for word in unknown_words:
        process_phrase(fst_file, [word], [word], {'PassThrough': 1}, weights, 0)
    # Mark starting state as final
    fst_file.write('0\n')
    fst_file.close()


def write_symbol_file(word_ids, symb_fname):
    symb_file = common.open_utf(symb_fname, 'w')
    for word in word_ids:
        symb_file.write(common.WORD_SYMB_TEMPLATE.format(word, word_ids[word]))
    symb_file.close()


def create_word_ids(words):
    word_ids = dict()
    # Add epsilon
    word_id = 0
    word_ids[common.EPSILON] = word_id
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
    pt_file = common.open_utf(pt_fname, 'r')
    for line in pt_file:
        phrase = line.strip(string.whitespace).split(' ||| ')
        src_words = phrase[1].strip(string.whitespace).split()
        tgt_words = phrase[2].strip(string.whitespace).split()
        features = collect_features(phrase[3], tgt_words)
        known_src_words.update(src_words)
        known_tgt_words.update(tgt_words)
        phrases.append((src_words, tgt_words, features))
    pt_file.close()

    unknown_words = set(sentence_words).difference(known_src_words)
    src_word_ids = create_word_ids(sentence_words)
    known_tgt_words.update(unknown_words)
    tgt_word_ids = create_word_ids(known_tgt_words)
    return phrases, unknown_words, src_word_ids, tgt_word_ids


def process_setence(sentence_str, pt_fname, weights, sentence_no, output_dir):
    # Make a set of all words in the source sentence
    sentence_words = sentence_str.strip(string.whitespace).split()
    # Read the phrase table file
    phrases, unknown_words, src_word_ids, tgt_word_ids = read_phrase_table(sentence_words, pt_fname)
    # Create file names
    fst_fname = common.make_path_name(output_dir, 'fst_txt', sentence_no)
    fst_bin_name = common.make_path_name(output_dir, 'fst_bin', sentence_no)
    isymb_fname = common.make_path_name(output_dir, 'isymb', sentence_no)
    osymb_fname = common.make_path_name(output_dir, 'osymb', sentence_no)
    # Write to files
    write_symbol_file(src_word_ids, isymb_fname)
    write_symbol_file(tgt_word_ids, osymb_fname)
    write_fst_file(phrases, unknown_words, weights, fst_fname)
    # Compile text files in binary fst
    common.make_fst(fst_fname, fst_bin_name, isymb_fname, osymb_fname)


def read_weights(weights_fname):
    weights = dict()
    weights_file = common.open_utf(weights_fname, 'r')
    for line in weights_file:
        line = line.strip(string.whitespace).split()
        if len(line) < 2:
            continue
        weights[line[0]] = float(line[1])
    weights_file.close()
    return weights

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

    pts = common.list_filter_sort_filenames(pts_dir, lambda f: os.path.isfile(f))

    # Open sentences and get ready to read
    sentences = common.open_utf(sentences_fname, 'r')
    sentence_no = 0
    for pt in pts:
        if sentence_no >= sentence_count:
            break
        sentence = sentences.readline()
        process_setence(sentence, pt, weights, sentence_no, output_dir)
        sentence_no += 1

    sentences.close()

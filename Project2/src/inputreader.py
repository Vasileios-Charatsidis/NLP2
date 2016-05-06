import os
import string
import math

import common


def read_sentences(source_sentences_fname, sentences_count):
    source_sentences = dict()
    source_sentences_file = common.open_utf(source_sentences_fname, 'r')
    sentence_no = 0
    for line in source_sentences_file:
        if sentence_no == sentences_count:
            break
        source_sentences[sentence_no] = line.strip(string.whitespace).split()
        sentence_no += 1
    source_sentences_file.close()
    return source_sentences


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


def parse_permutation_probability(metainfo):
    metainfo = metainfo.strip(string.whitespace).split()
    for feature in metainfo:
        feat = feature.split('=')
        if len(feat) < 2:
            continue
        if 'prob' == feat[0]:
            return float(feat[1])
    # Must have found prob in metainfo
    assert False


def read_permutations(permutations_fname, sentences_count):
    if permutations_fname is None:
        return None
    permutations_file = common.open_utf(permutations_fname, 'r')
    permutations_per_sentence = dict()
    for line in permutations_file:
        line = line.strip(string.whitespace).split(' ||| ')
        if len(line) < 4:
            continue
        sentence_no = int(line[0])
        if sentence_no == sentences_count:
            break
        permutation_prob = parse_permutation_probability(line[1])
        permutation = map(lambda x: int(x), line[2].strip(string.whitespace).split())
        permuted_words = line[3].strip(string.whitespace).split()
        if sentence_no not in permutations_per_sentence:
            permutations_per_sentence[sentence_no] = list()
        permutations_per_sentence[sentence_no].append((permutation_prob, permutation, permuted_words))
    permutations_file.close()
    return permutations_per_sentence


def read_input(source_sentences_fname, phrase_tables_dir, sentences_count, weights_fname, permutations_fname):
    # Read in sentences
    source_sentences = read_sentences(source_sentences_fname, sentences_count)
    # Read in the phrase_tables filenames
    phrase_table_fnames = common.list_filter_filenames(phrase_tables_dir, lambda f: os.path.isfile(f))
    # Read weights
    weights = read_weights(weights_fname)
    # Read permutations if present
    permutations_per_sentence = read_permutations(permutations_fname, sentences_count)
    return source_sentences, phrase_table_fnames, weights, permutations_per_sentence


def parse_features(features_str, tgt_words_count):
    features = features_str.strip(string.whitespace).split()
    feature_dict = dict()
    for feature in features:
        feat = feature.split('=')
        feature_dict[feat[0]] = float(feat[1])
    feature_dict['Glue'] = 1
    feature_dict['WordPenalty'] = - tgt_words_count / math.log(10)
    feature_dict['PassThrough'] = 0
    return feature_dict


def read_phrase_table(sentence, pt_fname):
    phrases = list()
    known_src_words = set()
    known_tgt_words = set()
    pt_file = common.open_utf(pt_fname, 'r')
    for line in pt_file:
        phrase = line.strip(string.whitespace).split(' ||| ')
        src_words = phrase[1].strip(string.whitespace).split()
        tgt_words = phrase[2].strip(string.whitespace).split()
        features = parse_features(phrase[3], len(tgt_words))
        known_src_words.update(src_words)
        known_tgt_words.update(tgt_words)
        phrases.append((src_words, tgt_words, features))
    pt_file.close()

    unknown_words = set(sentence).difference(known_src_words)
    known_tgt_words.update(unknown_words)
    tgt_vocab = common.extract_vocabulary(known_tgt_words)
    return phrases, unknown_words, tgt_vocab

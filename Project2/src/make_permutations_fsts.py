import sys
import os
import string
import common

error = 'Usage: make_permuations_fsts.py permutations_file output_dir sentence_count'


def write_permutation(fst_txt_file, permutation, permuted_words, prob, start_state, next_state):
    assert(len(permutation) == len(permuted_words))
    first = True
    for index, word in zip(permutation, permuted_words)[:-1]:
        if first:
            fst_txt_file.write(common.FST_TEMPLATE.format(start_state, next_state, index, word))
            first = False
        else:
            fst_txt_file.write(common.FST_TEMPLATE.format(next_state - 1, next_state, index, word))
        next_state += 1
    index = permutation[-1]
    word = permuted_words[-1]
    fst_txt_file.write(common.FST_WEIGHTED_TEMPLATE.format(next_state - 1, next_state, str(index), word, prob))
    fst_txt_file.write(str(next_state) + '\n')
    return next_state + 1


def extract_vocabulary(permutation, permuted_words):
    vocab = dict()
    for index, word in zip(permutation, permuted_words):
        vocab[index] = word
    return vocab


def make_symbol_mapping(symbols, self=False):
    mapping = dict()
    if self:
        for symbol in symbols:
            mapping[str(symbol)] = symbol+1
    else:
        for symbol in symbols:
            mapping[symbols[symbol]] = symbol+1
    return mapping


def write_symbol_file(symbol_mapping, symbols_fname):
    symbol_file = common.open_utf(symbols_fname, 'w')
    symbol_file.write(common.SYMB_TEMPLATE.format(common.EPSILON, 0))
    for symbol in symbol_mapping:
        symbol_file.write(common.SYMB_TEMPLATE.format(symbol, symbol_mapping[symbol]))
    symbol_file.close()


def write_symbol_files(vocab, isymb_fname, osymb_fname):
    insymbols = make_symbol_mapping(vocab.keys(), True)
    write_symbol_file(insymbols, isymb_fname)
    outsymbols = make_symbol_mapping(vocab)
    write_symbol_file(outsymbols, osymb_fname)


def encode_sentence_permutations(sentence_permutations, output_dir, sentence_no):
    # sentence_permuations is a list of triples: (prob, permutation, permuted_words)
    # if empty => nothing to do
    if not sentence_permutations:
        return
    start_state = 0
    next_state = 1
    fst_txt_fname = common.make_path_name(output_dir, 'fst_txt', sentence_no)
    fst_txt_file = common.open_utf(fst_txt_fname, 'w')
    for permutation in sentence_permutations:
        next_state = write_permutation(fst_txt_file, permutation[1], permutation[2], permutation[0], start_state, next_state)
    fst_txt_file.close()
    sentence_vocabulary = extract_vocabulary(sentence_permutations[0][1], sentence_permutations[0][2])
    isymb_fname = common.make_path_name(output_dir, 'isymb', sentence_no)
    osymb_fname = common.make_path_name(output_dir, 'osymb', sentence_no)
    write_symbol_files(sentence_vocabulary, isymb_fname, osymb_fname)
    fst_bin_name = common.make_path_name(output_dir, 'fst_bin', sentence_no)
    common.make_fst(fst_txt_fname, fst_bin_name, isymb_fname, osymb_fname)


def parse_permutation_probability(metainfo):
    metainfo = metainfo.strip(string.whitespace).split()
    for feature in metainfo:
        feat = feature.split('=')
        if 'prob' == feat[0]:
            return float(feat[1])
    # Must have found prob in metainfo
    assert False


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print error
        sys.exit()

    permutations_fname = sys.argv[1]
    output_dir = sys.argv[2]
    sentence_count = int(sys.argv[3])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    permutations_file = common.open_utf(permutations_fname, 'r')
    sentence_no = None
    sentence_permutations = list()
    for line in permutations_file:
        line = line.strip(string.whitespace).split(' ||| ')
        if len(line) < 4:
            continue
        if sentence_no is not None and sentence_no != int(line[0]):
            encode_sentence_permutations(sentence_permutations, output_dir, sentence_no)
            sentence_permutations = list()
        sentence_no = int(line[0])
        if sentence_no == sentence_count:
            break
        permutation_prob = parse_permutation_probability(line[1])
        permutation = map(lambda x: int(x), line[2].strip(string.whitespace).split())
        permuted_words = line[3].strip(string.whitespace).split()
        sentence_permutations.append((permutation_prob, permutation, permuted_words))

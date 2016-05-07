import os

import common
import inputreader as ir
import openfstio


def encode_sentences_to_fsts_monotone(sentences, output_dir):
    for sentence_no in sentences:
        sentence = sentences[sentence_no]
        # Create file names
        isymb_fname = common.make_path_name(output_dir, 'isymb', sentence_no)
        osymb_fname = common.make_path_name(output_dir, 'osymb', sentence_no)
        fst_txt_fname = common.make_path_name(output_dir, 'fst_txt', sentence_no)
        fst_bin_fname = common.make_path_name(output_dir, 'fst_bin', sentence_no)
        # Extract vocabulary
        vocab = common.extract_vocabulary(sentence)
        # Write symbol files
        openfstio.write_symbol_file(isymb_fname, vocab, True)
        openfstio.write_symbol_file(osymb_fname, vocab, False)
        openfstio.write_sentence(fst_txt_fname, sentence)
        # Compile text files in binary fst
        openfstio.make_fst(fst_txt_fname, fst_bin_fname, isymb_fname, osymb_fname)
        openfstio.sort_fst_arcs(fst_bin_fname, 'olabel')


def encode_sentences_to_fsts_lattice(sentences, permutations_per_sentence, weights, output_dir):
    weight = weights['LatticeCost']
    for sentence_no in sentences:
        sentence = sentences[sentence_no]
        sentence_permutations = permutations_per_sentence[sentence_no]
        # if empty => nothing to do
        if not sentence_permutations:
            return
        # Create file names
        isymb_fname = common.make_path_name(output_dir, 'isymb', sentence_no)
        osymb_fname = common.make_path_name(output_dir, 'osymb', sentence_no)
        fst_txt_fname = common.make_path_name(output_dir, 'fst_txt', sentence_no)
        fst_bin_fname = common.make_path_name(output_dir, 'fst_bin', sentence_no)
        # Extract vocabulary
        vocab = common.extract_vocabulary(sentence)
        # Write symbol files
        openfstio.write_symbol_file(isymb_fname, vocab, True)
        openfstio.write_symbol_file(osymb_fname, vocab, False)
        # Write fst text file
        start_state = 0
        next_state = 1
        fst_txt_file = common.open_utf(fst_txt_fname, 'w')
        for permutation in sentence_permutations:
            next_state = openfstio.write_permutation(fst_txt_file, permutation[1], permutation[2], permutation[0],
                                                     weight, start_state, next_state)
        fst_txt_file.close()
        openfstio.make_fst(fst_txt_fname, fst_bin_fname, isymb_fname, osymb_fname)
        openfstio.determinize_and_minimize(fst_bin_fname)
        openfstio.sort_fst_arcs(fst_bin_fname, 'olabel')


def encode_phrase_tables_to_fsts(sentences, phrase_table_fnames, weights, output_dir):
    for sentence_no in sentences:
        sentence = sentences[sentence_no]
        phrase_table_fname = phrase_table_fnames[sentence_no]
        phrases, unknown_words, tgt_vocab = ir.read_phrase_table(sentence, phrase_table_fname)
        src_vocab = common.extract_vocabulary(sentence)
        isymb_fname = common.make_path_name(output_dir, 'isymb', sentence_no)
        osymb_fname = common.make_path_name(output_dir, 'osymb', sentence_no)
        fst_txt_fname = common.make_path_name(output_dir, 'fst_txt', sentence_no)
        fst_bin_name = common.make_path_name(output_dir, 'fst_bin', sentence_no)
        # Write to files
        openfstio.write_symbol_file(isymb_fname, src_vocab, False)
        openfstio.write_symbol_file(osymb_fname, tgt_vocab, False)
        openfstio.write_pt_fst_file(fst_txt_fname, phrases, unknown_words, weights)
        # Compile text files in binary fst
        openfstio.make_fst(fst_txt_fname, fst_bin_name, isymb_fname, osymb_fname)
        openfstio.sort_fst_arcs(fst_bin_name, 'ilabel')


def make_translation_fsts(sentences_fsts_dir, phrase_table_fsts_dir, translation_type, derivations_count, output_dir):
    sentence_fst_fnames = common.list_filter_filenames(sentences_fsts_dir, lambda f: f.rfind('bin') > 0)
    pt_fst_fnames = common.list_filter_filenames(phrase_table_fsts_dir, lambda f: f.rfind('bin') > 0)
    for sentence_no in sentence_fst_fnames:
        sentence_fst_fname = sentence_fst_fnames[sentence_no]
        pt_fst_fname = pt_fst_fnames[sentence_no]
        translation_fst_fname = common.make_path_name(output_dir, 'fst_bin', sentence_no)
        openfstio.compose_fsts(sentence_fst_fname, pt_fst_fname, translation_fst_fname)
        best_fst_fname = common.make_path_name(output_dir, 'best_fst_bin', sentence_no)
        openfstio.get_best_fst(translation_fst_fname, best_fst_fname, derivations_count)
        best_derivations_fname = common.make_path_name(output_dir, 'best_derivations', sentence_no)
        openfstio.get_best_derivations(best_fst_fname, best_derivations_fname)
        # h stands for human readable
        best_derivations_h_fname = common.make_path_name(output_dir,
                                                         '{0}.{1:d}best'.format(translation_type, derivations_count),
                                                         sentence_no)
        openfstio.get_best_derivations_h(best_derivations_fname, best_derivations_h_fname)


def remove_alignments(derivation):
    words = derivation.split('|')[0::2]
    words = map(lambda word: word if 0 == len(word) or word[0] != ' ' else word[1:], words)
    translation = ''.join(words)
    return translation.strip()


def get_best_translation_with_best_derivation(best_derivations_fname):
    best_derivations_file = common.open_utf(best_derivations_fname, 'r')
    start_transitions, transitions, finals = openfstio.read_derivations(best_derivations_file)
    best_derivations_file.close()
    translations = dict()
    # len(start_transitions) > 0
    while start_transitions:
        start_transition = start_transitions.pop()
        translation, cost = openfstio.get_translation(start_transition, transitions, finals)
        derivation, _ = openfstio.get_translation(start_transition, transitions, finals, True)
        if translation not in translations:
            translations[translation] = list()
        translations[translation].append((cost, derivation))
    best_translation = None
    best_translation_cost = None
    best_derivation = None
    best_derivation_cost = None
    for translation in translations:
        derivations = translations[translation]
        translation_cost = 0
        for derivation in derivations:
            translation_cost += derivation[0]
            if best_derivation_cost is None or best_derivation_cost > derivation[0]:
                best_derivation_cost = derivation[0]
                best_derivation = derivation[1]
        if best_translation_cost is None or best_translation_cost > translation_cost:
            best_translation_cost = translation_cost
            best_translation = translation
    best_derivation = remove_alignments(best_derivation)
    return best_translation, best_derivation


def get_best_translations(translation_fsts_dir, translation_type, output_dir):
    derivations_fnames = common.list_filter_filenames(translation_fsts_dir, lambda f: f.rfind('derivations.') > 0)

    best_translations = common.open_utf(os.path.join(output_dir, translation_type+'.trans'), 'w')
    best_derivations = common.open_utf(os.path.join(output_dir, translation_type+'.der'), 'w')
    for sentence_no in derivations_fnames:
        derivations_fname = derivations_fnames[sentence_no]
        best_tr, best_der = get_best_translation_with_best_derivation(derivations_fname)
        best_translations.write(best_tr + '\n')
        best_derivations.write(best_der + '\n')
    best_translations.close()
    best_derivations.close()


def translate(source_sentences, phrase_table_fnames, weights, translation_type,
              permutations_per_sentence, derivations_count, output_dir):
    sentence_fsts_dir = os.path.join(output_dir, 'sentence_fsts')
    common.makedir(sentence_fsts_dir)
    if 'monotone' == translation_type:
        encode_sentences_to_fsts_monotone(source_sentences, sentence_fsts_dir)
    else:
        encode_sentences_to_fsts_lattice(source_sentences, permutations_per_sentence, weights, sentence_fsts_dir)
    phrase_table_fsts_dir = os.path.join(output_dir, 'phrase_table_fsts')
    common.makedir(phrase_table_fsts_dir)
    encode_phrase_tables_to_fsts(source_sentences, phrase_table_fnames, weights, phrase_table_fsts_dir)
    translation_fsts_dir = os.path.join(output_dir, 'translation_fsts')
    common.makedir(translation_fsts_dir)
    make_translation_fsts(sentence_fsts_dir, phrase_table_fsts_dir, translation_type,
                          derivations_count, translation_fsts_dir)
    translations_dir = os.path.join(output_dir, 'translations')
    common.makedir(translations_dir)
    get_best_translations(translation_fsts_dir, translation_type, translations_dir)

import sys
import gzip
import string
from translation import Translation


error = 'Usage: get_best_translations.py translations_file pos_tags_file weights_file out_best_file'


def calculate_score(weights, features):
    score = 0
    for feature_name in weights:
        assert(len(weights[feature_name]) == len(features[feature_name]))
        score += sum(map(lambda (x, y): x * y, zip(weights[feature_name], features[feature_name])))
    return score


def get_best_translation(translations, weights_dict):
    scores = map(lambda tr: calculate_score(weights_dict, tr.extract_features_with_names()), translations)
    return ' '.join(translations[scores.index(max(scores))].translation)


def write_best_translations(translations_fname, pos_tags_fname, weights_dict, out_fname):
    translations_file = gzip.open(translations_fname, 'r')
    pos_tags_file = open(pos_tags_fname, 'r')
    out_file = open(out_fname, 'w')

    translation_line = translations_file.readline()
    pos_tags_line = pos_tags_file.readline()
    sentence_translations = list()
    sentence_no = None
    while translation_line and pos_tags_line:
        translation_tokens = translation_line.strip(string.whitespace).split('|||')
        pos_tags_tokens = pos_tags_line.strip(string.whitespace).split('|||')
        assert(translation_tokens[0] == pos_tags_tokens[0])
        if not sentence_no:
            sentence_no = translation_tokens[0]
        elif sentence_no != translation_tokens[0]:
            best = get_best_translation(sentence_translations, weights_dict)
            out_file.write(best + '\n')
            sentence_translations = list()
            sentence_no = translation_tokens[0]
        sentence_translations.append(Translation(translation_tokens, '', pos_tags_tokens[1]))
        translation_line = translations_file.readline()
        pos_tags_line = pos_tags_file.readline()
    if sentence_translations:
        best = get_best_translation(sentence_translations, weights_dict)
        out_file.write(best + '\n')
        sentence_translations = list()
    translations_file.close()
    pos_tags_file.close()
    out_file.close()


def read_weights(weights_fname):
    weights = dict()
    weights_file = open(weights_fname, 'r')
    for line in weights_file:
        line = line.strip(string.whitespace).split('=')
        if len(line) < 2:
            continue
        weights[line[0]] = list()
        for weight in line[1].split():
            weights[line[0]].append(float(weight))
    weights_file.close()
    return weights


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print error
        sys.exit()

    translations_fname = sys.argv[1]
    pos_tags_fname = sys.argv[2]
    weights_fname = sys.argv[3]
    out_best_fname = sys.argv[4]

    weights_dict = read_weights(weights_fname)
    write_best_translations(translations_fname, pos_tags_fname, weights_dict, out_best_fname)


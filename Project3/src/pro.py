import sys
import random
from nltk.translate import bleu_score as bleu
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.svm import LinearSVC

from datareader import Reader

import time


error = 'Usage: python pro.py nbest_translations reference_translations src_sentence_prob pro_sample_size out_translations'


def extract_features(translation, tagger):
    features = translation.extract_features()
    return features


def subtract_lists(l1, l2):
    return map(lambda tup: tup[0] - tup[1], zip(l1, l2))


def multiply_lists(l1, l2):
    return reduce(lambda x, y: x + y[0]*y[1], zip(l1, l2), 0)


def evaluate_score(translation, score, smoothing_func):
    if score == 'BLEU':
        translation_split = translation.translation.split()
        reference_split = translation.reference.split()
        word_count = min(len(reference_split), len(translation_split))
        weights = []
        weight = 0.25
        if word_count < 4:
            weight = 1 / float(word_count)
        for i in range(min(4, word_count)):
            weights.append(weight)
        return bleu.sentence_bleu([reference_split], translation_split, weights=weights, smoothing_function=smoothing_func)
    else:
        print 'evaluate_score: unrecognized score \'{0}\''.format(score)


def sort_by(translations, score, smoothing_func):
    scored_translations = map(lambda translation: (evaluate_score(translation, score, smoothing_func), translation), translations)
    return sorted(scored_translations, reverse=True)


def pro(reader, sentence_prob, sample_size):
    start = time.time()
    training_data = []
    sf = bleu.SmoothingFunction()
    st = StanfordPOSTagger('stanford-postagger-full-2015-12-09/models/german-fast.tagger',
                           'stanford-postagger-full-2015-12-09/stanford-postagger.jar')
    nbest = True
    sentence_no = 0
    start1 = time.time()
    while nbest:
        sentence_no += 1
        if sentence_no % 100 == 0:
            print 'collected sentence tr data', sentence_no, time.time() - start1
            start1 = time.time()
        if random.random() > sentence_prob:
            nbest = reader.skip_next_src_nbest_translations()
            continue
        else:
            nbest = reader.read_next_src_nbest_translations()
        if not nbest:
            break
        sentence_training_data = list()
        while len(sentence_training_data) / 2 < sample_size:
            index1 = random.randint(0, len(nbest) - 1)
            index2 = random.randint(0, len(nbest) - 1)
            while index2 == index1:
                index2 = random.randint(0, len(nbest) - 1)
            translation1 = nbest[index1]
            translation2 = nbest[index2]
            sorted_translations = sort_by([translation1, translation2], 'BLEU', sf.method2)
            features = map(lambda translation: extract_features(translation[1], st), sorted_translations)
            for i in range(len(features)):
                instance = subtract_lists(features[i], features[(i + 1) % len(features)])
                label = 1 if sorted_translations[i][0] >= sorted_translations[(i + 1) % len(features)][0] else -1
                sentence_training_data.append((instance, label))
        training_data.extend(sentence_training_data)
    print 'collected training data', time.time() - start
    start = time.time()
    data, labels = zip(*training_data)
    print 'prepared training data', time.time() - start
    start = time.time()
    svc = LinearSVC()
    svc.fit(data, labels)
    print 'trained svm', time.time() - start
    return svc.coef_[0]


def get_best_translation(translations, weights):
    scores = map(lambda tr: multiply_lists(weights, tr.extract_features()), translations)
    return translations[scores.index(max(scores))].translation


def write_best_translations(reader, weights, out_fname):
    out_file = open(out_fname, 'w')
    nbest = reader.read_next_src_nbest_translations()
    sentence_no = 0
    while nbest:
        sentence_no += 1
        best = get_best_translation(nbest, weights)
        out_file.write(best.encode('utf-8'))
        out_file.write('\n')
        if sentence_no % 100 == 0:
            print 'wrote {0:d} translations'.format(sentence_no)
        nbest = reader.read_next_src_nbest_translations()
    out_file.close()


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print error
        sys.exit()

    translations_fname = sys.argv[1]
    references_fname = sys.argv[2]
    src_sentences_prob = float(sys.argv[3])
    pro_sample_size = int(sys.argv[4])
    out_translations_fname = sys.argv[5]

    nbest_reader = Reader(translations_fname, references_fname)
    weights = pro(nbest_reader, src_sentences_prob, pro_sample_size)
    nbest_reader.restart()
    write_best_translations(nbest_reader, weights, out_translations_fname)

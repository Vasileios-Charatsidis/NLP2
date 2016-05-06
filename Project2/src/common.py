import codecs
import os


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


def extract_vocabulary(sentence):
    vocab = dict()
    for index, word in enumerate(sentence):
        vocab[index] = word
    return vocab


def make_symbol_mapping(symbols, map_to_self=False):
    mapping = dict()
    if map_to_self:
        for symbol in symbols:
            mapping[str(symbol)] = symbol+1
    else:
        for symbol in symbols:
            mapping[symbols[symbol]] = symbol+1
    return mapping


def calculate_weight(features, weights):
    weight = 0
    for feature in features:
        weight += weights[feature] * features[feature]
    return weight

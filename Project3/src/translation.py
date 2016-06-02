import string


class Translation:

    def __init__(self, tokens, reference):
        self.features = self.parse_features(tokens[2].strip(string.whitespace))
        self.baseline_score = float(tokens[3].strip(string.whitespace))
        self.translation = self.parse_translation(tokens[1].strip(string.whitespace))
        self.reference = reference

    def parse_features(self, features_string):
        tokens = features_string.split()
        key = None
        features = dict()
        for token in tokens:
            if token[-1] == '=':
                key = token.strip().strip('=')
                features[key] = list()
            else:
                assert(key is not None)
                features[key].append(float(token.strip()))
        return features

    def parse_translation(self, phrases):
        phrases = phrases.split()
        translation = ''
        for phrase in phrases:
            if phrase[0] != '|':
                translation += phrase + ' '
        return translation.strip()

    def extract_features(self):
        features = list()
        for feature_name in self.features:
            features.extend(self.features[feature_name])
        return features

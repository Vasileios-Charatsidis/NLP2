import string


class Translation:

    def __init__(self, tokens, reference, pos_tags):
        self.features = self.parse_features(tokens[2].strip(string.whitespace))
        self.baseline_score = float(tokens[3].strip(string.whitespace))
        self.translation = self.parse_translation(tokens[1].strip(string.whitespace))
        self.reference = reference.split()
        self.pos_tags = pos_tags.strip(string.whitespace).split()

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
        translation = []
        phrases = phrases.split()
        for phrase in phrases:
            if phrase[0] != '|':
                translation.append(phrase)
        return translation

    def extract_features(self, basic):
        features = list()
        for feature_name in self.features:
            features.extend(self.features[feature_name])
        if basic:
            return features
        features.extend(self._extract_extra_features())
        features.append(self._has_verb())
        features.append(self._has_verb_at_end())
        features.append(self._has_verb_punctuation())
        return features

    def extract_feature_names(self, basic):
        feature_names = list()
        for feature_name in self.features:
            for _ in self.features[feature_name]:
                feature_names.append(feature_name)
        if basic:
            return feature_names
        feature_names.extend(['TranslationModelAvg', 'PermutationDistortionAvg'])
        feature_names.append('HasVerb')
        feature_names.append('HasVerbAtEnd')
        feature_names.append('HasVerbPunctAtEnd')
        return feature_names

    def extract_features_with_names(self):
        features_with_names = dict()
        for feature_name in self.features:
            features_with_names[feature_name] = self.features[feature_name]
        extra = self._extract_extra_features()
        features_with_names['TranslationModelAvg'] = [extra[0]]
        features_with_names['PermutationDistortionAvg'] = [extra[1]]
        features_with_names['HasVerb'] = [self._has_verb()]
        features_with_names['HasVerbAtEnd'] = [self._has_verb_at_end()]
        features_with_names['HasVerbPunctAtEnd'] = [self._has_verb_punctuation()]
        return features_with_names


    def _extract_extra_features(self):
        tr_model = self.features['TranslationModel0']
        perm_dist = self.features['PermutationDistortion0']
        features = [sum(tr_model)/float(len(tr_model)), sum(perm_dist) / float(len(perm_dist))]
        return features

    def _has_verb(self):
        for tag in self.pos_tags:
            if tag[0] == 'V':
                return 1
        return 0

    def _has_verb_at_end(self):
        if len(self.pos_tags) < 1:
            return 0
        return 1 if self.pos_tags[-1][0] == 'V' else 0

    def _has_verb_punctuation(self):
        if len(self.pos_tags) < 2:
            return 0
        if self.pos_tags[-1][0] == '$' and self.pos_tags[-2][0] == 'V':
            return 1
        return 0

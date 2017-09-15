from collections import namedtuple

import numpy as np

from sklearn.feature_extraction import DictVectorizer


Dataset = namedtuple('Dataset', ['inputs', 'targets'])


def extract_dummy_features(corpus):
    _, tids = tuple(zip(*corpus))
    return Dataset(inputs=np.zeros((len(corpus), 2)), targets=np.array(tids))


def extract_identity_features(corpus):
    wids, tids = tuple(zip(*corpus))
    return Dataset(inputs=np.array(wids).reshape(len(corpus), 1), targets=np.array(tids))


def extract_maxent_features(corpus, vocab):
    res = []
    tids = []
    for i, (wid, tid) in enumerate(corpus):
        tids.append(tid)
        features = {}
        word = vocab.get_word(wid)
        # Capitalization
        if _is_init_caps(word):
            features['initCaps'] = True
        elif word.isupper():
            features['allCaps'] = True
        elif any([c.isupper() for c in word[1:]]):
            features['mixedCaps'] = True
        # First word
        if i > 0 and corpus[i-1][0] == '.':
            if features.get('initCaps', False):
                features['firstWord-initCaps'] = True
            else:
                features['firstWord'] = True
        # Word length
        features['wordLen'] = len(word)
        # Punctuation
        if word[-1] == '.':
            features['endPeriod'] = True
        if '.' in word[:-1]:
            features['intPeriod'] = True
        if "'" in word:
            features['intQuote'] = True
        # Digits
        if word.isdigit():
            features['allDigits'] = True
        elif any([c.isdigit() for c in word]):
            features['intDigits'] = True
        # Contextual
        if i > 0 and corpus[i-1][0] in ['in', 'of']:
            features['w-1'] = corpus[i-1][0]
        if i + 1 < len(corpus) and corpus[i+1][0] in ['said', "'s"]:
            features['w+1'] = corpus[i+1][0]
        res.append(features)

    vec = DictVectorizer()
    return Dataset(inputs=vec.fit_transform(res), targets=np.array(tids))


def _is_init_caps(word):
    if not word:
        return False
    return word[0].isupper() and word[1:].islower()

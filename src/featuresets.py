def make_dummy_featuresets(corpus_reader):
    return [({}, tag) for _, tag in corpus_reader.tagged_words()]


def make_word_featuresets(corpus_reader):
    return [({'word': word}, tag) for word, tag in corpus_reader.tagged_words()]


# def extract_maxent_features(corpus, vocab):
#     res = []
#     tids = []
#     for i, (wid, tid) in enumerate(corpus):
#         tids.append(tid)
#         features = {}
#         word = vocab.get_word(wid)
#         # Capitalization
#         if _is_init_caps(word):
#             features['initCaps'] = True
#         elif word.isupper():
#             features['allCaps'] = True
#         elif any([c.isupper() for c in word[1:]]):
#             features['mixedCaps'] = True
#         # First word
#         if i > 0 and corpus[i-1][0] == '.':
#             if features.get('initCaps', False):
#                 features['firstWord-initCaps'] = True
#             else:
#                 features['firstWord'] = True
#         # Word length
#         features['wordLen'] = len(word)
#         # Punctuation
#         if word[-1] == '.':
#             features['endPeriod'] = True
#         if '.' in word[:-1]:
#             features['intPeriod'] = True
#         if "'" in word:
#             features['intQuote'] = True
#         # Digits
#         if word.isdigit():
#             features['allDigits'] = True
#         elif any([c.isdigit() for c in word]):
#             features['intDigits'] = True
#         # Contextual
#         if i > 0 and corpus[i-1][0] in ['in', 'of']:
#             features['w-1'] = corpus[i-1][0]
#         if i + 1 < len(corpus) and corpus[i+1][0] in ['said', "'s"]:
#             features['w+1'] = corpus[i+1][0]
#         res.append(features)

#     vec = DictVectorizer()
#     return Dataset(inputs=vec.fit_transform(res), targets=np.array(tids))


# def _is_init_caps(word):
#     if not word:
#         return False
#     return word[0].isupper() and word[1:].islower()

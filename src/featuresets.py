def make_dummy_featuresets(corpus_reader):
    return [({}, tag) for _, tag in corpus_reader.tagged_words()]


def make_word_featuresets(corpus_reader):
    return [({'word': word}, tag) for word, tag in corpus_reader.tagged_words()]


def make_maxent_featuresets(corpus_reader, ds=None):
    if ds is None:
        ds = range(-2, 3)
    featuresets = []
    tagged_words = corpus_reader.tagged_words()
    for i, (word, tag) in enumerate(tagged_words):
        fs = {}
        for d in ds:
            if not d:
                # Lexical
                fs['w'] = word
            elif i + d >= 0 and i + d < len(tagged_words):
                # Contextual
                key = f'w+{d}' if d > 0 else f'w{d}'
                fs[key] = tagged_words[i+d][0]
        featuresets.append((fs, tag))
    return featuresets

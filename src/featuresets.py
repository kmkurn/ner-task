def make_dummy_featuresets(corpus_reader):
    return [({}, tag) for _, tag in corpus_reader.tagged_words()]


def make_word_featuresets(corpus_reader):
    return [({'word': word}, tag) for word, tag in corpus_reader.tagged_words()]


def make_maxent_featuresets(corpus_reader):
    featuresets = []
    tagged_words = corpus_reader.tagged_words()
    for i, (word, tag) in enumerate(tagged_words):
        fs = {}
        # Lexical
        fs['w'] = word
        # Contextual
        if i - 2 >= 0:
            fs['w-2'] = tagged_words[i-2][0]
        if i - 1 >= 0:
            fs['w-1'] = tagged_words[i-1][0]
        if i + 1 < len(tagged_words):
            fs['w+1'] = tagged_words[i+1][0]
        if i + 2 < len(tagged_words):
            fs['w+2'] = tagged_words[i+2][0]
        featuresets.append((fs, tag))
    return featuresets

def make_dummy_featuresets(corpus_reader):
    return [({}, tag) for _, tag in corpus_reader.tagged_words()]


def make_word_featuresets(corpus_reader):
    return [({'word': word}, tag) for word, tag in corpus_reader.tagged_words()]


def make_maxent_featuresets(corpus_reader):
    featuresets = []
    tagged_words = corpus_reader.tagged_words()
    for i, (word, tag) in enumerate(tagged_words):
        fs = {}
        # Capitalization
        if _is_init_caps(word):
            fs['initCaps'] = True
        elif word.isupper():
            fs['allCaps'] = True
        elif any([c.isupper() for c in word[1:]]):
            fs['mixedCaps'] = True
        # First word
        if i > 0 and tagged_words[i-1][0] == '.':
            if fs.get('initCaps', False):
                fs['firstWord-initCaps'] = True
            else:
                fs['firstWord'] = True
        # Word length
        fs['wordLen'] = len(word)
        # Punctuation
        if word[-1] == '.':
            fs['endPeriod'] = True
        if '.' in word[:-1]:
            fs['intPeriod'] = True
        if "'" in word:
            fs['intQuote'] = True
        # Digits
        if word.isdigit():
            fs['allDigits'] = True
        elif any([c.isdigit() for c in word]):
            fs['intDigits'] = True
        # Contextual
        if i > 0 and tagged_words[i-1][0] in ['in', 'of']:
            fs['w-1'] = tagged_words[i-1][0]
        if i + 1 < len(tagged_words) and tagged_words[i+1][0] in ['said', "'s"]:
            fs['w+1'] = tagged_words[i+1][0]
        featuresets.append((fs, tag))
    return featuresets


def _is_init_caps(word):
    if not word:
        return False
    return word[0].isupper() and word[1:].islower()

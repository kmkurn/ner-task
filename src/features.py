def extract_dummy_features(corpus):
    return [((0, 0), pair.tag_id) for pair in corpus]


def extract_identity_features(corpus):
    return [((pair.word_id,), pair.tag_id) for pair in corpus]

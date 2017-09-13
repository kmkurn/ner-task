def extract_dummy_features(corpus):
    return [((0, 0), pair.tag_id) for pair in corpus]

def extract_dummy_features(corpus):
    return [((0, 0), tid) for _, tid in corpus]


def extract_identity_features(corpus):
    return [((wid,), tid) for wid, tid in corpus]

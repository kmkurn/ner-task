from collections import Counter

from nltk.classify.api import ClassifierI


class MajorityTag(ClassifierI):
    def __init__(self, majority_tag, tagset):
        self.majority_tag = majority_tag
        self.tagset = tagset

    def labels(self):
        return list(self.tagset)

    def classify(self, featureset):
        return self.majority_tag

    @classmethod
    def train(cls, train_toks):
        _, tags = tuple(zip(*train_toks))
        c = Counter(tags)
        return cls(c.most_common(1)[0][0], set(c.keys()))


class MemoTraining(ClassifierI):
    def __init__(self, memo, tagset):
        self.memo = memo
        self.tagset = tagset

    def labels(self):
        return list(self.tagset)

    def classify(self, featureset):
        return self.memo[featureset['word']]

    @classmethod
    def train(cls, train_toks):
        memo = {}
        tagset = set()
        for featureset, tag in train_toks:
            memo[featureset['word']] = tag
            tagset.add(tag)
        return cls(memo, tagset)

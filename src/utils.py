import numpy as np


class TermDict:
    def __init__(self, unk_token=None):
        self.unk_token = unk_token
        self._term2id = {}
        self._id2term = {}
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def get_id(self, term):
        try:
            return self._term2id[term]
        except KeyError:
            if self._frozen:
                if self.unk_token is None:
                    raise
                else:
                    self._add(self.unk_token)
                    return self._term2id[self.unk_token]
            else:
                self._add(term)
                return self._term2id[term]

    def get_term(self, term_id):
        return self._id2term[term_id]

    def _add(self, term):
        if term not in self._term2id:
            term_id = len(self._term2id)
            self._term2id[term] = term_id
            self._id2term[term_id] = term

    @property
    def terms(self):
        return self._term2id.keys()


class Dataset:
    def __init__(self, data):
        self.data = data
        self.inputs, self.targets = [], []
        for pair in data:
            self.inputs.append(pair[0])
            self.targets.append(pair[1])
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

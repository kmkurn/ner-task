from collections import namedtuple
import os

from src.corpus import WordTagPair
from src.utils import TermDict


WordTagIdPair = namedtuple('WordTagIdPair', ['word_id', 'tag_id'])


class Vocabulary:
    def __init__(self, unk_word_token=None, unk_tag_token=None):
        self._word_dict = TermDict(unk_token=unk_word_token)
        self._tag_dict = TermDict(unk_token=unk_tag_token)

    def freeze(self):
        self._word_dict.freeze()
        self._tag_dict.freeze()

    def get_word_id(self, word):
        return self._word_dict.get_id(word)

    def get_tag_id(self, tag):
        return self._tag_dict.get_id(tag)

    def get_word(self, word_id):
        return self._word_dict.get_term(word_id)

    def get_tag(self, tag_id):
        return self._tag_dict.get_term(tag_id)

    def fit(self, corpus):
        for pair in corpus:
            self.get_word_id(pair.word)
            self.get_tag_id(pair.tag)
        self.freeze()

    def transform(self, corpus):
        res = []
        for pair in corpus:
            wid = self.get_word_id(pair.word)
            tid = self.get_tag_id(pair.tag)
            res.append(WordTagIdPair(wid, tid))
        return res

    def fit_transform(self, corpus):
        res = self.transform(corpus)
        self.freeze()
        return res

    def inverse_transform(self, corpus):
        res = []
        for pair in corpus:
            word = self.get_word(pair.word_id)
            tag = self.get_tag(pair.tag_id)
            res.append(WordTagPair(word, tag))
        return res

    def save_to_dir(self, output_dir):
        # Write word dict to file
        with open(os.path.join(output_dir, 'words.tsv'), 'w') as f:
            for word in self.words:
                print(f'{word}\t{self.get_word_id(word)}', file=f)
        # Write tag dict to file
        with open(os.path.join(output_dir, 'tags.tsv'), 'w') as f:
            for tag in self.tags:
                print(f'{tag}\t{self.get_tag_id(tag)}', file=f)

    @property
    def words(self):
        return self._word_dict.terms

    @property
    def tags(self):
        return self._tag_dict.terms

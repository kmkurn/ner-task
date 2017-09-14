from collections import Counter
import os

from src.utils import TermDict


class Vocabulary:
    def __init__(self, unk_word_token=None, unk_tag_token=None, min_word_count=1):
        self.min_word_count = min_word_count
        self._word_dict = TermDict(unk_token=unk_word_token)
        self._tag_dict = TermDict(unk_token=unk_tag_token)

    def get_word_id(self, word):
        return self._word_dict.get_id(word)

    def get_tag_id(self, tag):
        return self._tag_dict.get_id(tag)

    def get_word(self, word_id):
        return self._word_dict.get_term(word_id)

    def get_tag(self, tag_id):
        return self._tag_dict.get_term(tag_id)

    def fit(self, corpus):
        words, _ = tuple(zip(*corpus))
        self.counter = Counter(words)
        for word, tag in corpus:
            if self.counter[word] > self.min_word_count:
                self._word_dict.add(word)
                self._tag_dict.add(tag)

    def transform(self, corpus):
        return [(self.get_word_id(word), self.get_tag_id(tag)) for word, tag in corpus]

    def inverse_transform(self, corpus):
        return [(self.get_word(wid), self.get_tag(tid)) for wid, tid in corpus]

    def save_to_dir(self, output_dir):
        # Write word dict to file
        with open(os.path.join(output_dir, 'vocab-words.tsv'), 'w') as f:
            for word in self.words:
                print(f'{word}\t{self.get_word_id(word)}', file=f)
        # Write tag dict to file
        with open(os.path.join(output_dir, 'vocab-tags.tsv'), 'w') as f:
            for tag in self.tags:
                print(f'{tag}\t{self.get_tag_id(tag)}', file=f)

    @property
    def words(self):
        return self._word_dict.terms

    @property
    def tags(self):
        return self._tag_dict.terms

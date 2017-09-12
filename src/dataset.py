from argparse import ArgumentParser
from collections import defaultdict
import csv
import os
import random
import sys

from src.utils import TermDict


class WordTagPair:
    __slots__ = ['word', 'tag']

    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def __str__(self):
        return f'{self.word}/{self.tag}'

    def __repr__(self):
        return f'{type(self).__name__}({self.word!r}, {self.tag!r})'


class Sentence(list):
    def __str__(self):
        return '  '.join([str(p) for p in self])


class CoNLL:
    def __init__(self, corpus_dir, which='train', load=True):
        if which not in ['train', 'dev']:
            raise ValueError(f"'which' can only be 'train' or 'dev', got '{which}'")

        self.corpus_dir = corpus_dir
        self.which = which
        self._sentences = []
        self._tag_index = defaultdict(list)
        if load:
            self.load()

    def __iter__(self):
        return iter(self._sentences)

    def load(self):
        self._sentences = []
        self._tag_index = defaultdict(list)

        corpus_path = os.path.join(self.corpus_dir, f'{self.which}.conll')
        with open(corpus_path, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            tmp = Sentence()
            for row in reader:
                if row:
                    pair = WordTagPair(row[0], row[1])
                    self._tag_index[pair.tag].append(pair.word)
                    tmp.append(pair)
                else:
                    self._sentences.append(tmp)
                    tmp = Sentence()
            if tmp:
                self._sentences.append(tmp)

    def summarize(self):
        n_words = sum(len(words) for words in self._tag_index.values())

        print('The corpus has:')
        print(f'{len(self._sentences)} sentences')
        print(f'{n_words} words')
        for tag, words in self._tag_index.items():
            print(f'{len(words)} words tagged with {tag}')

    def sample_sentences(self, size=10):
        return random.sample(self._sentences, size)

    def sample_words(self, tag='O', size=10):
        return random.sample(self._tag_index[tag], size)


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

    @property
    def words(self):
        return self._word_dict.terms

    @property
    def tags(self):
        return self._tag_dict.terms


class CoNLLDataset:
    def __init__(self, train_conll, dev_conll):
        self.train_conll = train_conll
        self.dev_conll = dev_conll

        self.fetch_all_sets()

    def fetch_all_sets(self):
        self._vocab = Vocabulary(unk_word_token='UNK')
        self._X_train, self._y_train = self._fetch_from(self.train_conll)
        self._vocab.freeze()
        self._X_dev, self._y_dev = self._fetch_from(self.dev_conll)

    def _fetch_from(self, conll):
        wids, tids = [], []
        for sent in conll:
            for pair in sent:
                wids.append(self._vocab.get_word_id(pair.word))
                tids.append(self._vocab.get_tag_id(pair.tag))
        return wids, tids

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def train_set(self):
        return (self._X_train, self._y_train)

    @property
    def dev_set(self):
        return (self._X_dev, self._y_dev)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for interacting with the dataset')
    parser.add_argument('corpus_dir', help='corpus directory')
    parser.add_argument('--which', choices=['train', 'dev'], default='train',
                        help='which dataset to load (default: train)')
    parser.add_argument('command', help='command to execute', choices=['summarize', 'sample'])
    parser.add_argument('--sentence', '-s', action='store_true', help='sample sentences')
    parser.add_argument('--words', '-w', action='store_true', help='sample words')
    parser.add_argument('--tag', '-t', default='O', choices=['O', 'MISC', 'ORG', 'PER', 'LOC'],
                        help='sample only words with this tag')
    parser.add_argument('--size', '-n', type=int, default=10, help='sample size')
    args = parser.parse_args()

    conll = CoNLL(args.corpus_dir, which=args.which)

    if args.command == 'summarize':
        conll.summarize()
    elif args.command == 'sample':
        if args.sentence:
            for sent in conll.sample_sentences(size=args.size):
                print(sent)
        elif args.words:
            print(f'Words with {args.tag} tag:')
            for word in conll.sample_words(tag=args.tag, size=args.size):
                print(word)
        else:
            print('Error: can only sample words or sentences', file=sys.stderr)
            sys.exit(1)

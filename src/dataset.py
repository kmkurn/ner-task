from argparse import ArgumentParser
from collections import defaultdict
import csv
import os
import random
import sys


class WordTagPair:
    __slots__ = ['word', 'tag']

    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def __str__(self):
        return f'{self.word}/{self.tag}'


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
        n_words = 0
        for words in self._tag_index.values():
            n_words += len(words)

        print('The corpus has:')
        print(f'{len(self._sentences)} sentences')
        print(f'{n_words} words')
        for tag, words in self._tag_index.items():
            print(f'{len(words)} words tagged with {tag}')

    def sample_sentences(self, size=10):
        return random.sample(self._sentences, size)

    def sample_words(self, tag='O', size=10):
        return random.sample(self._tag_index[tag], size)


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

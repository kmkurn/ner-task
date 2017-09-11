from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import csv
import os


WordTagPair = namedtuple('WordTagPair', ['word', 'tag'])


class CoNLL:
    def __init__(self, corpus_dir, which='train', load=True):
        if which not in ['train', 'dev']:
            raise ValueError(f"'which' can only be 'train' or 'dev', got '{which}'")

        self.corpus_dir = corpus_dir
        self.which = which
        self._sentences = []
        if load:
            self.load()

    def load(self):
        corpus_path = os.path.join(self.corpus_dir, f'{self.which}.conll')
        with open(corpus_path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            tmp = []
            for row in reader:
                if row:
                    tmp.append(WordTagPair(row[0], row[1]))
                else:
                    self._sentences.append(tmp)
                    tmp = []

    def summarize(self):
        n_sents = 0
        n_words = 0
        tag_counts = defaultdict(int)

        for sent in self._sentences:
            n_sents += 1
            n_words += len(sent)
            for pair in sent:
                tag_counts[pair.tag] += 1

        print('The corpus has:')
        print(f'{n_sents} sentences')
        print(f'{n_words} words')
        for tag, count in tag_counts.items():
            print(f'{count} words tagged with {tag}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for interacting with the dataset')
    parser.add_argument('corpus_dir', help='corpus directory')
    parser.add_argument('--which', '-w', choices=['train', 'dev'], default='train',
                        help='which dataset to load (default: train)')
    parser.add_argument('command', help='command to execute', choices=['summarize'])
    args = parser.parse_args()

    conll = CoNLL(args.corpus_dir, which=args.which)

    if args.command == 'summarize':
        conll.summarize()

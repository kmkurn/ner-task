from argparse import ArgumentParser
from collections import defaultdict
import csv
import random
import sys

from src.utils import WordTagPair


class CoNLLCorpus:
    DOCSTART_TOKEN = '-DOCSTART-'

    def __init__(self, corpus_path, strip_docstarts=True):
        self.corpus_path = corpus_path
        self.strip_docstarts = strip_docstarts
        self._sentences = []
        self._tag_index = defaultdict(list)
        self.load()

    def __iter__(self):
        return iter(self._sentences)

    def flatten(self):
        return [pair for sent in self for pair in sent]

    def load(self):
        self._sentences = []
        self._tag_index = defaultdict(list)

        with open(self.corpus_path, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            tmp_sent = []
            for row in reader:
                if row:
                    if self.strip_docstarts and row[0] == self.DOCSTART_TOKEN:
                        continue
                    pair = WordTagPair(row[0], row[1])
                    self._tag_index[pair.tag].append(pair.word)
                    tmp_sent.append(pair)
                elif tmp_sent:
                    self._sentences.append(tmp_sent)
                    tmp_sent = []
            if tmp_sent:
                self._sentences.append(tmp_sent)

    def summarize(self, file=None):
        if file is None:
            file = sys.stdout

        n_words = sum(len(words) for words in self._tag_index.values())

        print('The corpus has:', file=file)
        print(f'{len(self._sentences)} sentences', file=file)
        print(f'{n_words} word tokenss', file=file)
        for tag, words in self._tag_index.items():
            print(f'{len(words)} word tokens tagged with {tag}', file=file)

    def sample_sentences(self, size=10):
        return random.sample(self._sentences, size)

    def sample_words(self, tag='O', size=10):
        return random.sample(self._tag_index[tag], size)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for interacting with the given CoNLL corpus')
    parser.add_argument('command', help='command to execute',
                        choices=['summarize', 'sample', 'print'])
    parser.add_argument('corpus_file', help='path to corpus file')
    parser.add_argument('--strip-docstarts', action='store_true', default=True,
                        help='whether to strip -DOCSTART- elements (default: True)')
    parser.add_argument('--sentence', '-s', action='store_true', default=True,
                        help='sample sentences (default: True)')
    parser.add_argument('--words', '-w', action='store_true', help='sample words')
    parser.add_argument('--tag', '-t', default='O', choices=['O', 'MISC', 'ORG', 'PER', 'LOC'],
                        help='sample only words with this tag (default: O)')
    parser.add_argument('--size', '-n', type=int, default=10, help='sample size (default: 10)')
    parser.add_argument('--strip-blank-lines', action='store_true', default=True,
                        help='whether to strip blank lines when printing (default: True)')
    args = parser.parse_args()

    corpus = CoNLLCorpus(args.corpus_file, strip_docstarts=args.strip_docstarts)

    if args.command == 'sample':
        if args.words:
            print(f'Words with {args.tag} tag:')
            for i, word in enumerate(corpus.sample_words(tag=args.tag, size=args.size)):
                print(f'{i+1})', end=' ')
                print(word)
        else:
            for i, sent in enumerate(corpus.sample_sentences(size=args.size)):
                print(f'{i+1})', end=' ')
                print('  '.join([f'{pair.word}/{pair.tag}' for pair in sent]))
    elif args.command == 'print':
        for sent in corpus:
            for pair in sent:
                print(f'{pair.word}\t{pair.tag}')
            if not args.strip_blank_lines:
                print()
    else:
        corpus.summarize()

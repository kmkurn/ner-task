from argparse import ArgumentParser
from collections import defaultdict
import os
import random

from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.tokenize import RegexpTokenizer, BlanklineTokenizer


DOCSTART_WORD = '-DOCSTART-'
DOCSTART_TAG = 'O'


def read_docstart_delimited_block(stream):
    s = []
    while True:
        line = stream.readline()
        # End of file:
        if not line:
            if s:
                return [''.join(s)]
            else:
                return []
        # DOCSTART line:
        elif line.startswith(DOCSTART_WORD):
            stream.readline()  # discard next line b/c it's blank
            if s:
                del s[-1]  # discard previous line b/c it's also blank
                return [''.join(s)]
        # Other line:
        else:
            s.append(line)


class CoNLLCorpus:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        corpus_dir, corpus_file = os.path.split(corpus_path)
        self.reader = TaggedCorpusReader(
            corpus_dir, [corpus_file], sep='\t',
            word_tokenizer=RegexpTokenizer(r'\n', gaps=True),
            sent_tokenizer=BlanklineTokenizer(),
            para_block_reader=read_docstart_delimited_block)

    def summarize(self):
        n_paras = len(self.reader.paras())
        n_sents = len(self.reader.sents())
        n_words = len(self.reader.words())
        counter = defaultdict(int)
        for _, tag in self.reader.tagged_words():
            counter[tag] += 1

        out = [
            f'{n_paras} paragraphs',
            f'{n_sents} sentences',
            f'{n_words} word tokens'
        ]
        for tag, counts in counter.items():
            out.append(f'{counts} word tokens tagged with {tag}')
        return '\n'.join(out)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for interacting with the given CoNLL corpus')
    parser.add_argument('command', help='command to execute',
                        choices=['summarize', 'sample'])
    parser.add_argument('corpus_file', help='path to corpus file')
    parser.add_argument('--sentence', '-s', action='store_true', default=True,
                        help='sample sentences (default: True)')
    parser.add_argument('--words', '-w', action='store_true', help='sample words')
    parser.add_argument('--tag', '-t', default='O', choices=['O', 'MISC', 'ORG', 'PER', 'LOC'],
                        help='sample only words with this tag (default: O)')
    parser.add_argument('--size', '-n', type=int, default=10, help='sample size (default: 10)')
    args = parser.parse_args()

    corpus = CoNLLCorpus(args.corpus_file)

    if args.command == 'sample':
        if args.words:
            population = [word for word, tag_ in corpus.reader.tagged_words()
                          if tag_ == args.tag]
            print(f'Words with {args.tag} tag:')
            for i, word in enumerate(random.sample(population, args.size)):
                print(f'{i+1})', end=' ')
                print(word)
        else:
            population = list(corpus.reader.tagged_sents())
            for i, sent in enumerate(random.sample(population, args.size)):
                print(f'{i+1})', end=' ')
                print('  '.join([f'{word}/{tag}' for word, tag in sent]))
    else:
        print('The corpus has:')
        print(corpus.summarize())

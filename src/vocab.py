from argparse import ArgumentParser
from collections import Counter

from src.corpus import DOCSTART_WORD, DOCSTART_TAG, CoNLLCorpus


class Vocabulary:
    def __init__(self, unk_word_token, min_word_count=2):
        self.unk_word_token = unk_word_token
        self.min_word_count = min_word_count
        self.word_counts = {}
        self.wordset = set()
        self.tagset = set()

    def train(self, corpus_reader):
        words, tags = tuple(zip(*corpus_reader.tagged_words()))
        self.word_counts = Counter(words)
        self.wordset = {word for word, count in self.word_counts.items()
                        if count >= self.min_word_count}
        self.tagset = set(tags)


if __name__ == '__main__':
    parser = ArgumentParser(description='Unkify words in a corpus given a training corpus')
    parser.add_argument('train_file', help='path to training corpus')
    parser.add_argument('corpus_file', help='path to corpus to be unkified')
    parser.add_argument('--unk-token', '-u', default='-UNK-',
                        help='special token for unknown words (default: -UNK-)')
    parser.add_argument('--min-count', '-c', type=int, default=2,
                        help='min count a word should have to be included in the vocabulary'
                        ' (default: 2)')
    args = parser.parse_args()

    train = CoNLLCorpus(args.train_file)
    corpus = CoNLLCorpus(args.corpus_file)
    vocab = Vocabulary(args.unk_token, min_word_count=args.min_count)
    vocab.train(train.reader)

    out = []
    for tagged_para in corpus.reader.tagged_paras():
        out.append(f'{DOCSTART_WORD}\t{DOCSTART_TAG}')
        out.append('')
        for tagged_sent in tagged_para:
            for word, tag in tagged_sent:
                if word in vocab.wordset:
                    out.append(f'{word}\t{tag}')
                else:
                    out.append(f'{args.unk_token}\t{tag}')
            out.append('')
    print('\n'.join(out))

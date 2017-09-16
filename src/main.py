from argparse import ArgumentParser
import sys
import pickle

from nltk.classify.maxent import MaxentClassifier, TypedMaxentFeatureEncoding

from src.corpus import CoNLLCorpus, DOCSTART_WORD, DOCSTART_TAG
from src.evaluation import evaluate, pretty_format
from src.featuresets import (make_dummy_featuresets, make_word_featuresets,
                             make_maxent_featuresets)
from src.models import MajorityTag, MemoTraining


if __name__ == '__main__':
    parser = ArgumentParser(description='The main script to run NER models')
    parser.add_argument('--model-name', '-n', choices=['majority', 'memo', 'maxent'],
                        required=True, help='model name')
    parser.add_argument('--corpus', '-c', required=True, help='path to corpus file')
    parser.add_argument('--model-path', '-m', required=True,
                        help='path to save/load the trained model')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='whether to do training or testing/inference (default: train)')
    args = parser.parse_args()

    print('COMMAND:', ' '.join(sys.argv), file=sys.stderr, end='\n\n')

    if args.mode == 'train':
        train_corpus = CoNLLCorpus(args.corpus)
        out = train_corpus.summarize().split('\n')
        print('* Loaded training corpus', file=sys.stderr, end='\n  ')
        print('\n  '.join(out), file=sys.stderr)

        print('* Training model...', end=' ', file=sys.stderr)
        if args.model_name == 'memo':
            train_toks = make_word_featuresets(train_corpus.reader)
            model = MemoTraining.train(train_toks)
        elif args.model_name == 'maxent':
            train_toks = make_maxent_featuresets(train_corpus.reader)
            encoding = TypedMaxentFeatureEncoding.train(train_toks)
            model = MaxentClassifier.train(train_toks, encoding=encoding, min_lldelta=0.001)
        else:
            train_toks = make_dummy_featuresets(train_corpus.reader)
            model = MajorityTag.train(train_toks)
        print('done', file=sys.stderr)

        with open(args.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'* Model saved to {args.model_path}', file=sys.stderr)

        print(f'* Evaluate on training set:', file=sys.stderr)
        train_featuresets = [fs for fs, _ in train_toks]
        hyp_tags = model.classify_many(train_featuresets)
        ref_tags = [tag for _, tag in train_corpus.reader.tagged_words()]
        result = evaluate(ref_tags, hyp_tags)
        print(pretty_format(result), file=sys.stderr)
    else:
        dev_corpus = CoNLLCorpus(args.corpus)
        out = dev_corpus.summarize().split('\n')
        print('* Loaded dev corpus', file=sys.stderr, end='\n  ')
        print('\n  '.join(out), file=sys.stderr)

        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
        print(f'* Model loaded from {args.model_path}', file=sys.stderr)
        print(f'* Evaluate on dev set:', file=sys.stderr)
        if args.model_name == 'memo':
            featuresets = make_word_featuresets(dev_corpus.reader)
        else:
            featuresets = make_dummy_featuresets(dev_corpus.reader)
        dev_featuresets = [fs for fs, _ in featuresets]
        hyp_tags = model.classify_many(dev_featuresets)
        ref_tags = [tag for _, tag in dev_corpus.reader.tagged_words()]
        result = evaluate(ref_tags, hyp_tags)
        print(pretty_format(result), file=sys.stderr)

        # Print hyp corpus to stdout
        out, i = [], 0
        for para in dev_corpus.reader.paras():
            out.append(f'{DOCSTART_WORD}\t{DOCSTART_TAG}')
            out.append('')
            for sent in para:
                for word in sent:
                    tag = hyp_tags[i]
                    out.append(f'{word}\t{tag}')
                    i += 1
                out.append('')
        print('\n'.join(out))

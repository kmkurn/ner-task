from argparse import ArgumentParser
import sys

from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib

from src.corpus import CoNLLCorpus
from src.evaluation import evaluate
from src.features import extract_dummy_features, extract_identity_features
from src.models import MemorizeTrainingClassifier
from src.vocab import Vocabulary
from src.utils import Dataset


if __name__ == '__main__':
    parser = ArgumentParser(description='The main script to run NER models')
    parser.add_argument('--model-name', '-n', choices=['majority', 'memo'], required=True,
                        help='model name')
    parser.add_argument('--train-corpus', '-t', required=True, help='path to training corpus')
    parser.add_argument('--dev-corpus', '-d', required=True, help='path to dev corpus')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='whether to do training or testing/inference (default: train)')
    parser.add_argument('--model-path', '-m', required=True,
                        help='path to save/load the trained model')
    parser.add_argument('--strip-docstarts', action='store_true', default=True,
                        help='whether to strip -DOCSTART- elements (default: True)')
    parser.add_argument('--min-count', '-c', default=1, type=int,
                        help='min word count for rare words (default: 1)')
    args = parser.parse_args()

    print('COMMAND:', ' '.join(sys.argv), file=sys.stderr)
    train_corpus = CoNLLCorpus(args.train_corpus, strip_docstarts=args.strip_docstarts)
    print('* Loaded training corpus.', file=sys.stderr, end=' ')
    train_corpus.summarize(file=sys.stderr)
    dev_corpus = CoNLLCorpus(args.dev_corpus, strip_docstarts=args.strip_docstarts)
    print('* Loaded dev corpus.', file=sys.stderr, end=' ')
    dev_corpus.summarize(file=sys.stderr)
    vocab = Vocabulary(unk_word_token='-UNK-', min_word_count=args.min_count)
    vocab.fit(train_corpus.flatten())
    print(f'* Built vocabulary containing {len(vocab.words)} word types', file=sys.stderr)
    train_corpus = vocab.transform(train_corpus.flatten())
    dev_corpus = vocab.transform(dev_corpus.flatten())

    if args.model_name == 'majority':
        train_data = extract_dummy_features(train_corpus)
        dev_data = extract_dummy_features(dev_corpus)
    else:
        train_data = extract_identity_features(train_corpus)
        dev_data = extract_identity_features(dev_corpus)

    train_set = Dataset(train_data)
    dev_set = Dataset(dev_data)

    if args.mode == 'train':
        if args.model_name == 'majority':
            clf = DummyClassifier(strategy='most_frequent')
        else:
            clf = MemorizeTrainingClassifier()
        print('* Training model...', end=' ', file=sys.stderr)
        clf.fit(train_set.inputs, train_set.targets)
        print('done', file=sys.stderr)

        train_outputs = clf.predict(train_set.inputs)
        train_hyps = [(vocab.get_word(wid), vocab.get_tag(output_tid))
                      for (wid, _), output_tid in zip(train_corpus, train_outputs)]
        train_refs = vocab.inverse_transform(train_corpus)
        train_f1 = evaluate(train_refs, train_hyps, metric='f1')
        print('* Training F1 scores:', file=sys.stderr)
        for tag in sorted(train_f1.keys()):
            f1_score = train_f1[tag]
            print(f'    {tag}: {f1_score:.2f}', file=sys.stderr)

        joblib.dump(clf, args.model_path)
        print(f'* Model saved to {args.model_path}', file=sys.stderr)
    else:
        clf = joblib.load(args.model_path)
        print(f'* Model loaded from {args.model_path}', file=sys.stderr)
        dev_outputs = clf.predict(dev_set.inputs)
        dev_hyps = [(vocab.get_word(wid), vocab.get_tag(output_tid))
                    for (wid, _), output_tid in zip(dev_corpus, dev_outputs)]
        dev_refs = vocab.inverse_transform(dev_corpus)
        dev_f1 = evaluate(dev_refs, dev_hyps, metric='f1')
        print('* Dev F1 scores:', file=sys.stderr)
        for tag in sorted(dev_f1.keys()):
            f1_score = dev_f1[tag]
            print(f'    {tag}: {f1_score:.2f}', file=sys.stderr)

        result = []
        for (wid, tid), output_tid in zip(dev_corpus, dev_outputs):
            result.append((wid, output_tid))
        result = vocab.inverse_transform(result)
        for word, tag in result:
            print(f'{word}\t{tag}')

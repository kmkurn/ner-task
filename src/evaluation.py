from argparse import ArgumentParser
import sys

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate(refs, hyps, file=None, overall=False):
    if len(refs) != len(hyps):
        raise ValueError('reference and hypothesis has different number of lines')

    ref_words, ref_tags = tuple(zip(*refs))
    hyp_words, hyp_tags = tuple(zip(*hyps))

    for (ref_word, ref_tag), (hyp_word, hyp_tag) in zip(refs, hyps):
        if ref_tag != hyp_tag and file is not None:
            print(f'Wrong tag: {ref_word:20}\t{hyp_word:20}\t{ref_tag:5}\t{hyp_tag:5}',
                  file=file)

    unique_tags = list(sorted(set(ref_tags)))
    average = 'macro' if overall else None
    precision = precision_score(ref_tags, hyp_tags, labels=unique_tags, average=average)
    recall = recall_score(ref_tags, hyp_tags, labels=unique_tags, average=average)
    f1 = f1_score(ref_tags, hyp_tags, labels=unique_tags, average=average)
    cm = confusion_matrix(ref_tags, hyp_tags, labels=unique_tags)

    if overall:
        return precision, recall, f1
    else:
        return dict(zip(unique_tags, zip(precision, recall, f1))), cm


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate a hypothesis against a reference')
    parser.add_argument('reference', help='path to reference file')
    parser.add_argument('hypothesis', help='path to hypothesis file')
    args = parser.parse_args()

    with open(args.reference) as f:
        refs = [(l.strip().split()[0], l.strip().split()[1]) for l in f if l.strip()]
    with open(args.hypothesis) as f:
        hyps = [(l.strip().split()[0], l.strip().split()[1]) for l in f if l.strip()]

    result, cm = evaluate(refs, hyps, file=sys.stderr)
    for tag in sorted(result.keys()):
        value = result[tag]
        print(f'{tag:5}:', end=' ')
        if isinstance(value, tuple):
            print('prec={:.2f} recall={:.2f} f1={:.2f}'.format(*value))
        else:
            print(f'{value:.2f}')
    print(cm)

from argparse import ArgumentParser
from collections import defaultdict
import sys


def evaluate(ref_path, hyp_path, metric='all', file=None):
    if metric not in ['all', 'precision', 'recall', 'f1']:
        raise ValueError(
            f"'metric' can only be 'all', 'precision', 'recall', or 'f1'; got '{metric}'")

    with open(ref_path) as f:
        ref_lines = [line.strip() for line in f if line.strip()]
    with open(hyp_path) as f:
        hyp_lines = [line.strip() for line in f if line.strip()]

    if len(ref_lines) != len(hyp_lines):
        raise ValueError('reference and hypothesis has different number of lines')

    ref_count = defaultdict(int)
    hyp_count = defaultdict(int)
    true_pos = defaultdict(int)
    for i, (ref_line, hyp_line) in enumerate(zip(ref_lines, hyp_lines)):
        ref_word, ref_tag = ref_line.split()
        hyp_word, hyp_tag = hyp_line.split()

        ref_count[ref_tag] += 1
        hyp_count[hyp_tag] += 1
        if ref_tag == hyp_tag:
            true_pos[ref_tag] += 1
        elif file is not None:
            print(f'Wrong tag: {ref_word:20}\t{hyp_word:20}\t{ref_tag:5}\t{hyp_tag:5}',
                  file=file)

    res = {}
    for tag in ref_count:
        prec = true_pos[tag] / hyp_count[tag] if hyp_count[tag] != 0 else 0.
        recall = true_pos[tag] / ref_count[tag]
        f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0. else 0.
        if metric == 'all':
            res[tag] = (prec, recall, f1)
        elif metric == 'precision':
            res[tag] = prec
        elif metric == 'recall':
            res[tag] = recall
        else:
            res[tag] = f1

    return res


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate a hypothesis against a reference')
    parser.add_argument('reference', help='path to reference file')
    parser.add_argument('hypothesis', help='path to hypothesis file')
    parser.add_argument('--metric', '-m', choices=['all', 'precision', 'recall', 'f1'],
                        default='all', help='metric to use in evaluation (default: all)')
    args = parser.parse_args()

    result = evaluate(args.reference, args.hypothesis, metric=args.metric, file=sys.stderr)
    for tag, value in result.items():
        print(f'{tag}:', end=' ')
        if isinstance(value, tuple):
            print('prec={:.2f} recall={:.2f} f1={:.2f}'.format(*value))
        else:
            print(f'{value:.2f}')

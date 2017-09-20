from argparse import ArgumentParser
from collections import defaultdict, Counter, namedtuple
import sys
from warnings import warn

import matplotlib.pyplot as plt
from nltk.metrics.scores import precision, recall, f_measure
from nltk.metrics.confusionmatrix import ConfusionMatrix
import numpy as np

from src.corpus import CoNLLCorpus


OVERALL_KEY = '-OVERALL-'
EvalResult = namedtuple('EvalResult', ['precision', 'recall', 'f1', 'conf_matrix'])

plt.style.use('ggplot')


def evaluate(ref_tags, hyp_tags):
    if len(ref_tags) != len(hyp_tags):
        raise ValueError('reference and hypothesis has different number of lines')

    n = len(ref_tags)
    counter = Counter(ref_tags)
    unique_tags = set(ref_tags)
    prec_dict, rec_dict, f_dict = defaultdict(float), defaultdict(float), defaultdict(float)
    for tag in sorted(unique_tags):
        ref_ids = {i for i, ref_tag in enumerate(ref_tags) if ref_tag == tag}
        hyp_ids = {i for i, hyp_tag in enumerate(hyp_tags) if hyp_tag == tag}
        prec_dict[tag] = precision(ref_ids, hyp_ids)
        rec_dict[tag] = recall(ref_ids, hyp_ids)
        f_dict[tag] = f_measure(ref_ids, hyp_ids)
        if prec_dict[tag] is None:
            warn(f'Undefined precision for {tag}; converting to 0.0')
            prec_dict[tag] = 0.
        if rec_dict[tag] is None:
            warn(f'Undefined recall for {tag}; converting to 0.0')
            rec_dict[tag] = 0.
        if f_dict[tag] is None:
            warn(f'Undefined F-score for {tag}; converting to 0.0')
            f_dict[tag] = 0.
        prec_dict[OVERALL_KEY] += counter[tag] * prec_dict[tag] / n
        rec_dict[OVERALL_KEY] += counter[tag] * rec_dict[tag] / n
        f_dict[OVERALL_KEY] += counter[tag] * f_dict[tag] / n

    return EvalResult(precision=prec_dict, recall=rec_dict, f1=f_dict,
                      conf_matrix=ConfusionMatrix(ref_tags, hyp_tags, sort_by_count=True))


def report_tag_mismatch(refs, hyps):
    out = ['List of tag mismatch errors:']
    max_len = max(len(word) for word, _ in refs)
    fmt = f'{{:{max_len}}}\t{{}}\t{{}}'
    for (ref_word, ref_tag), (_, hyp_tag) in zip(refs, hyps):
        if ref_tag != hyp_tag:
            out.append(fmt.format(ref_word, ref_tag, hyp_tag))
    return '\n'.join(out)


def pretty_format(result):
    out = result.conf_matrix.pretty_format().split('\n')
    for key in sorted(result.precision.keys()):
        prec, rec, f1 = result.precision[key], result.recall[key], result.f1[key]
        out.append(f'{key:10}: prec={prec:.2f} recall={rec:.2f} f1={f1:.2f}')
    return '\n'.join(out)


def plot_conf_matrix(conf_matrix, ref_tags, filename):
    conf_arr = [[conf_matrix[t1, t2] for t2 in ref_tags] for t1 in ref_tags]
    n = len(ref_tags)

    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i)
        for j in i:
            tmp_arr.append(j / a)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.grid(False)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    for x in range(n):
        for y in range(n):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    fig.colorbar(res)
    plt.xticks(range(n), ref_tags)
    plt.yticks(range(n), ref_tags)
    plt.savefig(filename)


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate a hypothesis against a reference')
    parser.add_argument('reference', help='path to reference file')
    parser.add_argument('hypothesis', help='path to hypothesis file')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='whether to report tag mismatch errors to stderr')
    parser.add_argument('--save-cm-to', help='plot confusion matrix and save it as this file')
    args = parser.parse_args()

    ref_corpus = CoNLLCorpus(args.reference)
    hyp_corpus = CoNLLCorpus(args.hypothesis)

    ref_tags = [tag for _, tag in ref_corpus.reader.tagged_words()]
    hyp_tags = [tag for _, tag in hyp_corpus.reader.tagged_words()]
    result = evaluate(ref_tags, hyp_tags)
    print(pretty_format(result))

    if args.verbose:
        print(report_tag_mismatch(ref_corpus.reader.tagged_words(),
                                  hyp_corpus.reader.tagged_words()), file=sys.stderr)
    if args.save_cm_to is not None:
        plot_conf_matrix(result.conf_matrix, sorted(set(ref_tags)), args.save_cm_to)

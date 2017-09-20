# Maximum Entropy Model for Named Entity Recognition

This is an implementation of a simple maximum entropy model for named entity recognition (NER) task.

## Requirements

- Python 3.6
- [NLTK](http://nltk.org/) version 3.2 or newer

To enable plotting, the following packages are also needed

- [Matplotlib](http://matplotlib.org/) version 2.0 or newer
- [Numpy](http://numpy.org/) version 1.13 or newer

Once all dependencies are installed, you'll also need to install this package in editable mode.

    git clone https://github.com/kmkurn/ner-task.git  # clone this repository
    cd ner-task  # move into this project directory
    pip install -e .  # install the package in editable mode

## How to use

### Corpus

One corpus that can be used is [CoNLL corpus](https://drive.google.com/file/d/0BwUacZglWV28VGJCNm1pTk5Hbmc/view). This corpus has two files, `train.conll` and `dev.conll`. The first 20 lines from `train.conll` are

    -DOCSTART-	O

    EU	ORG
    rejects	O
    German	MISC
    call	O
    to	O
    boycott	O
    British	MISC
    lamb	O
    .	O

    Peter	PER
    Blackburn	PER

    BRUSSELS	LOC
    1996-08-22	O

    The	O
    European	ORG

As we can see, each word is in its own line with its tag, separated by a tab (`\t`) character. Each sentence is separated by a blank line and each document is separated by a special line with `-DOCSTART-` as the word. Any corpus that are compatible with this format can be used.

### Corpus summary and sampling

Script `src/corpus.py` provides functionality to print corpus summary and perform sampling (sentences or words having a certain tag). To print corpus summary, use

    python src/corpus.py summarize [corpus file, e.g. train.conll]

This will print statistics of the corpus file like number of sentences, words, etc. To sample sentences, use

    python src/corpus.py --size 5 sample [corpus file]

This will sample 5 sentences from the corpus file. To sample words instead, use

    python src/corpus.py --size 5 -w sample [corpus file]

By default, this will sample only words with tag `O`. To specify another tag, use `-t` option. For more info, run `python src/corpus.py -h`

### Unkification

File `src/vocab.py` can be used to unkify the corpus.

    python src/vocab.py train.conll train.conll > train.conll.unk
    python src/vocab.py train.conll dev.conll > dev.conll.unk

The first argument is the training file from which the vocabulary will be built. The second argument is the corpus file to unkify. All words that are not contained in the vocabulary will be converted into a special UNK token. By default, this token is `-UNK-`. Specify another token with `--unk-token [UNK token]` option. Also, by default, only words that occur at least twice in the training file that are included in the vocabulary. To change this, use `--min-count [cutoff]`. As usual, more info can be viewed by running `python src/vocab.py -h`.

### Training

Model training is provided by `src/main.py` script. The full usage of this script is

    usage: main.py [-h] --model-name {majority,memo,maxent} --corpus CORPUS
                  --model-path MODEL_PATH [--mode {train,test}] [--cutoff CUTOFF]
                  [--max-iter MAX_ITER] [--contexts [CONTEXTS [CONTEXTS ...]]]

    The main script to run NER models

    optional arguments:
      -h, --help            show this help message and exit
      --model-name {majority,memo,maxent}, -n {majority,memo,maxent}
                            model name
      --corpus CORPUS, -c CORPUS
                            path to corpus file
      --model-path MODEL_PATH, -m MODEL_PATH
                            path to save/load the trained model
      --mode {train,test}   whether to do training or testing/inference (default:
                            train)
      --cutoff CUTOFF       feature count cutoff for maxent (default: 2)
      --max-iter MAX_ITER   max number of training iteration for maxent (default:
                            50)
      --contexts [CONTEXTS [CONTEXTS ...]]
                            contexts to include as features for maxent (default:
                            -2 -1 0 1 2)

To train the baseline model (which only memorizes word-tag assignment in the training data), run

    python src/main.py -n memo -c train.conll.unk -m memo-model.pkl --mode train > train-memo.log 2>&1

This will save the trained model to `memo-model.pkl` file and the training log to `train-memo.log`. Similarly, training the maximum entropy model can be done by

    python src/main.py -n maxent -c train.conll.unk -m maxent-model.pkl --mode train > train-maxent.log 2>&1

By default, this will use 5 features (current word, two words before, and two words after). Option `--contexts` can be used to specify which words to include as features. Options `--cutoff` and `--max-iter` can be used to specify minimum number of feature occurrence to be included in the model (features occurring fewer than the cutoff will be discarded) and the number of iterations when training respectively.

### Evaluation

To evaluate the model against a development/testing set, the same `src/main.py` script can be used. As an example

    python src/main.py -n maxent -c dev.conll.unk -m maxent-model.pkl --mode test > output-maxent.conll 2> test-maxent.log

This will write the predicted tags of the words in `dev.conll.unk` file to `output-maxent.conll` in the same tab-delimited format and the log messages to `test-maxent.log`. The log file will also contain the precision, recall, F1 score, and the confusion matrix of the model on the development set.

A more complete evaluation is provided by `src/evaluation.script`. As an illustration

    python src/evaluation.py -v dev.conll.unk output-maxent.conll > report-maxent.out 2> report-maxent.err

This will output the scores (like in `test-maxent.log`) to `report-maxent.out` and a list of words that are misclassified (along with the true and predicted tag) to `report-maxent.err`. To plot the confusion matrix and save it to a file, use `--save-cm-to [filename]` option. Run `python src/evaluation.py -h` for more info.

## License

This software is licensed with the MIT license. See `LICENSE.txt` for the full text.

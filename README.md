# Autumn NER

A standalone supervised sequence-to-sequence deep learning system for Named Entity Recognition (NER). 

At the character level, word representations are composed via character CNNs (window size=3) over each *character* embedding vector and its corresponding *character type* (lowercase, uppercase, punctation, etc) embedding vector. The CNN-based word representations are further concatenated to a corresponding *word* embedding vector and *word type* embedding vector. Finally, the word representations are passed into a bi-directional LSTM layer. The output of the LSTM unit at each timestep is fully-connected to its own separate softmax output layer so a prediction label ([IOB format](https://en.wikipedia.org/wiki/Inside_Outside_Beginning)) can be made for each token. Here the final loss for an example is the mean loss over all output layers.

An advantage of this system is that it does not utilize an NLP pipeline (e.g. tagging, chunking, then NER). Here we train on IOB labels and predict them directly.

~~~
numpy
sklearn
tensorflow 1.0.0 with tensorflow-fold
~~~

# Training

The `--datapath` option must point to a directory where training files can be found. The program will look in particular for files ending with `train.ner.txt`, `dev.ner.txt`, and `test.ner.txt`. It will train on `train.ner.txt` files and tune on `dev.ner.txt`. 

## Training on the CoNL2003 Dataset

For a simple no-frills training of the CoNLL2003 dataset, run the following command.

`python train.py --datapath=CoNLL2003`

The model is stored in a created directory named "saved_model".

## Using pre-trained embeddings

By default, the model will not use pre-trained word embeddings. To use pre-trained word embeddings, utilize the `--embeddings` option. To use [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for example, run:

`python train.py --datapath=CoNLL2003 --embeddings=glove.6B.300d.txt`

## Training options

Default parameters:

~~~
batch_size=64
datapath='CoNLL2003'
decay_rate=0.95
embedding_factor=1.0  # comparative learning rate for the embedddings
embeddings_path=None  # if None, embeddings are randomly initialized
keep_prob=0.7 # probability to keep for dropout
learning_rate='default'  # default=0.001, or 0.5 if adagrad
num_cores=5
num_epoch=50
optimizer='default' # default=rmsprop, other options: adam, adagrad
seed=1
~~~

Here are the rest of the options:

~~~
usage: train.py [-h] [--datapath DATAPATH] [--embeddings EMBEDDINGS_PATH]
                [--optimizer OPTIMIZER] [--batch-size BATCH_SIZE]
                [--num-epoch NUM_EPOCH] [--learning-rate LEARNING_RATE]
                [--embedding-factor EMBEDDING_FACTOR] [--decay DECAY_RATE]
                [--keep-prob KEEP_PROB] [--num-cores NUM_CORES] [--seed SEED]

Train and evaluate BiLSTM on a given dataset

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   path to the datasets
  --embeddings EMBEDDINGS_PATH
                        path to the testing dataset
  --optimizer OPTIMIZER
                        choose the optimizer: default, rmsprop, adagrad, adam.
  --batch-size BATCH_SIZE
                        number of instances in a minibatch
  --num-epoch NUM_EPOCH
                        number of passes over the training set
  --learning-rate LEARNING_RATE
                        learning rate
  --embedding-factor EMBEDDING_FACTOR
                        learning rate multiplier for embeddings
  --decay DECAY_RATE    exponential decay for learning rate
  --keep-prob KEEP_PROB
                        dropout keep rate
  --num-cores NUM_CORES
                        seed for training
  --seed SEED           seed for training

~~~

# Annotating

To annotate an unseen example, run `annotate.py` and pass the input through STDIN. The annotator expects one sentence per line. As an example:

`echo "Peter Minuit is credited with the purchase of the island of Manhattan in 1626." | python annotate.py`

The corresponding output:

```
Peter I-PER
Minuit I-PER
is O
credited O
with O
the O
purchase O
of O
the O
island O
of O
Manhattan I-LOC
in O
1626. O
```

Or alternatively, pass the input in as a file:

`python annotate.py <input file>`

# Testing

The `test.py` is mostly only used to evaluate the model. Like `train.py` it will train on the `train.ner.txt` and tune on the `dev.ner.txt`, but it will additionally report separate evaluations on each `test.ner.txt` file. The commandline options are the same as that of `train.py`. This does not store the model for use by the `annotate.py` module.

`python test.py --datapath=CoNLL2003`

# Acknowledgements

This implementation is heavily inspired by the model proposed in the following paper:

> Chiu, Jason PC, and Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs." Transactions of the Association for Computational Linguistics 4 (2016): 357-370. [Preprint](https://arxiv.org/pdf/1511.08308.pdf)

The evaluation script is from the following repository and is released under the MIT license. The original script is slightly modified to be easily imported directly from the training component.

https://github.com/spyysalo/conlleval.py

# Author

> Tung Tran  
> tung.tran **[at]** uky.edu  
> <http://tttran.net/>



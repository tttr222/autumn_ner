#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, fileinput
import numpy as np
from model import AutumnNER
import sklearn.metrics as skm
import argparse
import conlleval

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')

def main(args):
    print >> sys.stderr, "Running Autumn NER model annotating module"
    print >> sys.stderr, args
    random.seed(args.seed)
    
    print >> sys.stderr, "Loading vocab.."
    with open(os.path.join('saved_model','word_vocab.pickle'),'r') as f:
        word_vocab = pickle.load(f)

    print >> sys.stderr, "Loading labels.."
    with open(os.path.join('saved_model','label_space.pickle'),'r') as f:
        labels = pickle.load(f)
    
    
    # Create the model, passing in relevant parameters
    bilstm = AutumnNER(labels=labels,
                    word_vocab=word_vocab,
                    word_embeddings=None)
    
    bilstm.restore('./saved_model/main')
    
    inputs = []
    for line in fileinput.input(args.files):
        line = line.rstrip()
        if len(line) == 0:
            continue
        
        items = line.split(' ')
        inputs.append([ x.strip() for x in items if len(x.strip()) > 0])
    
    for line, tag in zip(inputs, bilstm.predict(inputs,batch_size=20)):
        for pair in zip(line, tag):
            print ' '.join(pair)
        
        print ''

def report_performance(model, X_test,y_test, outname):
    micro_evaluation = model.evaluate(X_test,y_test,macro=False)
    macro_evaluation = model.evaluate(X_test,y_test,macro=True)
    print "Micro Test Eval: F={:.4f} P={:.4f} R={:.4f}".format(*micro_evaluation)
    print "Macro Test Eval: F={:.4f} P={:.4f} R={:.4f}".format(*macro_evaluation)
    
    pred_test = model.predict(X_test)
    
    with open(outname,'w') as f:
        for (x,y,z) in zip(X_test,y_test,pred_test):
            for token, y_true, y_pred in zip(x,y,z):
                print >> f, token, y_true, y_pred
            
            print >> f, ''
    
    with open(outname,'r') as f:
        evaluation.report(evaluation.evaluate(f))

def load_embeddings(fname, vocab, dim=200):
    cached = 'scratch/embeddings_{}.npy'.format(abs(hash(' '.join(vocab))))
    
    if not os.path.exists(cached):
        weight_matrix = np.random.uniform(-0.05, 0.05, (len(vocab),dim)).astype(np.float32)
        ct = 0
        with codecs.open(fname, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(u' ', 1)
                if word not in vocab:
                    continue
                
                idx = vocab.index(word)
                vec = np.array(vec.split(), dtype=np.float32)
                weight_matrix[idx,:dim] = vec[:dim]
                ct += 1
                if ct % 33 == 0:
                    sys.stdout.write('Loading embeddings {}/{}   \r'.format(ct, len(vocab)))
        
        print "Loaded {}/{} embedding vectors".format(ct, len(vocab))
        np.save(cached,weight_matrix)
    else:
        weight_matrix = np.load(cached)
    
    print >> sys.stderr, "Loaded weight matrix {}..".format(weight_matrix.shape)
    
    return weight_matrix


def load_dataset(fname, shuffle=False):
    dataset = []
    with open(fname,'r') as f:
        dataset = [ x.split('\n') for x in f.read().split('\n\n') if x ]
    
    vocab = []
    output = []
    for x in dataset:
        
        tokens, labels = zip(*[ z.split(' ') for z in x if z ])
        for t in tokens:
            t = t.lower()
            if t not in vocab:
                vocab.append(t)
        
        output.append((tokens, labels))
    
    return output, vocab

if __name__ == '__main__':
    main(parser.parse_args())

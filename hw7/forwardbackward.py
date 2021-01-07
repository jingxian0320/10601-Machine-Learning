#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:54:43 2020

@author: jingxianbao
"""


import sys
import numpy as np

def read_matrix(path):
    data = []
    with open(path) as csv_file:
        lines = csv_file.read().split('\n')
        line_count = 0
        for row in lines:
            if row == '':
                continue
            new_row = []
            elements = row.split(" ")
            for e in elements:
                new_row.append(float(e))
            data.append(new_row)
            line_count += 1
    print('Processed %d lines.'%line_count)
    return np.array(data)


def read_csv(path):
    data = []
    with open(path) as csv_file:
        lines = csv_file.read().split('\n')
        line_count = 0
        for row in lines:
            if row == '':
                continue
            new_row = []
            elements = row.split(" ")
            for e in elements:
                word, tag = e.split("_")
                new_row.append([word, tag])
            data.append(new_row)
            line_count += 1
    print('Processed %d lines.'%line_count)
    return data


def to_dict(path):
    word2ind = {}
    ind2word = {}
    with open(path) as csv_file:
        lines = csv_file.read().split('\n')
        line_count = 0
        for word in lines:
            if word == '':
                continue
            word2ind[word] = line_count
            ind2word[line_count] = word
            line_count += 1
    print('Processed %d lines.'%line_count)
    return word2ind, ind2word


def log_exp_sum(x):
    m = np.max(x)
    return np.log(np.sum(np.exp(x - m))) + m
    
    
def forwardbackward(data, prior, trans, emit, n_word, n_tag): #one row
    T = len(data)
    alpha = np.zeros((n_tag, T))
    beta = np.zeros((n_tag, T))
    prior = np.log(prior)
    trans = np.log(trans)
    emit = np.log(emit)
    alpha[:, [0]] = emit[:,[data[0][0]]] + prior
    for t in range(1, T):
        for k in range(0, n_tag):
            temp = []
            for j in range (n_tag):
                temp.append(emit[k, data[t][0]] + alpha[j, t-1] + trans[j, k])
            alpha[k, t] += log_exp_sum(temp) 
    for t in range(T-2, -1, -1):
        for k in range(0, n_tag):
            temp = []
            for j in range (n_tag):
                temp.append(emit[j,data[t+1][0]] + beta[j, t+1] + trans[k, j])
            beta[k, t] += log_exp_sum(temp)
    
    ll = log_exp_sum(alpha[:, -1])
    
    # predict
    predict = np.argmax(alpha + beta, axis = 0)
    return ll, predict
        
def evaluate(data, prior, trans, emit, ind2word, ind2tag, word2ind, tag2ind):
    n_word = len(word2ind.keys())
    n_tag = len(tag2ind.keys())  
    lls = []
    predictions = []
    n = 0
    correct = 0
    for i in range(len(data)):
        data_in_ind = [(word2ind[word], tag2ind[tag]) for word, tag in data[i]]
        ll, prediction = forwardbackward(data_in_ind, prior, trans, emit, n_word, n_tag)
        lls.append(ll)
        # accuracy
        label = [x[1] for x in data_in_ind]
        for l, p in zip(label, list(prediction)):
            n += 1
            if l == p:
                correct += 1
        
        # map to word
        prediction = [(x[0][0], ind2tag[x[1]]) for x in zip(data[i], prediction)]
        predictions.append(prediction)
    return np.mean(lls), correct/n, predictions
    

def output_metrics(ll, accuracy, out_file):
    with open(out_file, "w") as output:
        output.write("Average Log-Likelihood: {0:.6f}\n".format(ll))
        output.write("Accuracy: {0:.6f}".format(accuracy))
        
        
def output_predicted(predicted, out_file):
    with open(out_file, "w") as output:
        for row in predicted:
            s = []
            for e in row:
                s.append(e[0]+'_'+e[1])
            output.write(' '.join(s) +'\n') 
        
        
if __name__ == '__main__':
    test_in = sys.argv[1]
    ind_to_word = sys.argv[2]
    ind_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]
    
    test_data = read_csv(test_in)
    prior = read_matrix(hmmprior)
    trans = read_matrix(hmmtrans)
    emit = read_matrix(hmmemit)
    
    word2ind, ind2word = to_dict(ind_to_word)
    tag2ind, ind2tag = to_dict(ind_to_tag)
    
    ll, accuracy, predictions = evaluate(test_data, prior, trans, emit, ind2word, ind2tag, word2ind, tag2ind)
    print (ll)
    output_metrics(ll, accuracy, metric_file)
    output_predicted(predictions, predicted_file)
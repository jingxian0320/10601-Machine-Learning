#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:54:30 2020

@author: jingxianbao
"""

import sys
import numpy as np

def read_csv(path, n = None):
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
    if n != None:
        return data[:n]
    return data

def to_dict(path):
    d = {}
    with open(path) as csv_file:
        lines = csv_file.read().split('\n')
        line_count = 0
        for row in lines:
            if row == '':
                continue
            d[row] = line_count
            line_count += 1
    print('Processed %d lines.'%line_count)
    return d
    
def process(data, word_dict, tag_dict):
    n_word = len(word_dict.keys())
    n_tag = len(tag_dict.keys())
    prior = np.ones((n_tag,1))
    trans = np.ones((n_tag, n_tag))
    emit = np.ones((n_tag, n_word))
    for line in data:
        head_tag_ind = tag_dict[line[0][1]]
        prior[head_tag_ind, 0] += 1
        for i in range(len(line)):
            tag_ind = tag_dict[line[i][1]]
            word_ind = word_dict[line[i][0]]
            emit[tag_ind, word_ind] += 1
            if i != 0:
                prev_tag_ind = tag_dict[line[i-1][1]]
                trans[prev_tag_ind,tag_ind] += 1
    prior = prior/np.sum(prior, axis = 0)
    trans = trans/np.sum(trans, axis = 1, keepdims = True)
    emit = emit/np.sum(emit, axis = 1, keepdims = True)
    return prior, trans, emit
            

def output(matrix, out_file):
    with open(out_file, "w") as output:
        for row in matrix:
             output.write(' '.join([str(x) for x in row])+'\n')
               

if __name__ == '__main__':
    train_in = sys.argv[1]
    ind_to_word = sys.argv[2]
    ind_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    
    data = read_csv(train_in)
    word_dict = to_dict(ind_to_word)
    tag_dict = to_dict(ind_to_tag)
    prior, trans, emit = process(data, word_dict, tag_dict)
    output(prior, hmmprior)
    output(emit, hmmemit)
    output(trans, hmmtrans)
    
    
    
    

    
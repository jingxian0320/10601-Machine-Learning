#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:10:36 2020

@author: jingxianbao
"""

import sys


def read_csv(path):
    labels = []
    words = []
    with open(path) as csv_file:
        csv_reader = csv_file.readlines()
        line_count = 0
        for row in csv_reader:
            row = row.split("\t")
            labels.append(row[0])
            words.append(row[1].split(" "))
            line_count += 1
    print('Processed %d lines.'%line_count)
    return {"labels":labels, "words":words}


def read_dict(path):
    word_dict = {}
    with open(path) as csv_file:
        csv_reader = csv_file.readlines()
        for row in csv_reader:
            row = row.split(" ")
            word_dict[row[0]] = int(row[1]) #word to idx
    return word_dict

def create_feature_count(data, word_dict):
    out_data = []
    for i in range(len(data['labels'])):
        label = data['labels'][i]
        words = data['words'][i]
        new_out_data = {}
        new_out_data['label'] = label
        new_out_data['features'] = {}
        for word in words:
            if word in word_dict:
                idx = word_dict[word]
                if idx not in new_out_data['features']:
                    new_out_data['features'][idx] = 1
                else:
                    new_out_data['features'][idx] += 1
        out_data.append(new_out_data)
    return out_data

def output_feature(out_path, out_data, feature_flag):
    with open(out_path, "w") as output:
        for line in out_data:
            out_list = []
            out_list.append(line['label'])
            for k,v in line['features'].items():
                if feature_flag == 2:
                    if v >= 4:
                        continue
                out_list.append(str(k)+":1")        
            output.write("\t".join(out_list)+"\n")
        

    
if __name__ == '__main__':
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    val_out = sys.argv[6]
    test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    
    train_data = read_csv(train_in)
    val_data = read_csv(val_in)
    test_data = read_csv(test_in)
    word_dict = read_dict(dict_in)
    
    out_train_data = create_feature_count(train_data, word_dict)
    out_val_data = create_feature_count(val_data, word_dict)
    out_test_data = create_feature_count(test_data, word_dict)
    
    output_feature(train_out, out_train_data, feature_flag)
    output_feature(val_out, out_val_data, feature_flag)
    output_feature(test_out, out_test_data, feature_flag)
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:29:03 2020

@author: jingxianbao
"""

import sys
import csv



def read_csv(path = "politicians_train.tsv"):
    print (path)
    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data.append(row)
                line_count += 1
    print('Processed %d lines.'%line_count)
    return data

def fit(data,split_ind = 0):
    count = {}
    for row in data:
        x = row[split_ind]
        if x not in count:
            count[x] = {}
        y = row[-1]
        if y not in count[x]:
            count[x][y] = 1
        else:
            count[x][y] += 1
        model = {}
        for split in count.keys():
            decision = sorted(list(count[split].items()),key=lambda x:x[1],reverse=True)[0][0]
            model[split] = decision
    return [split_ind, model]

def predict(model, data):
    predicted = []
    split_ind = model[0]
    model = model[1]
    for row in data:
        x = row[split_ind]
        predicted.append(model[x])
    return predicted

def evaluate(data, predict):
    total = len(data)
    error = 0
    for i in range(total):
        if data[i][-1] != predict[i]:
            error += 1
    return error/total
        
def output_metrics(train_err, test_err, out_file):
    with open(out_file, "w") as output:
        output.write("error(train): {0:.6f}\n".format(train_err))
        output.write("error(test): {0:.6f}".format(test_err))

def output_predicted(predicted, out_file):
    with open(out_file, "w") as output:
        for result in predicted:
            output.write(result+'\n')
            
if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    split_ind = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    print("The train input file is:%s" % (train_in))
    print("The test input file is:%s" % (test_in))
    print("The split index is:%s" % (split_ind))
    print("The train output file is:%s" % (train_out))
    print("The test output file is:%s" % (test_out))
    print("The metrics out file is:%s" % (metrics_out))
    
    train_data = read_csv(train_in)
    test_data = read_csv(test_in)
    model = fit(train_data, split_ind)
    train_predict = predict(model, train_data)
    test_predict = predict(model, test_data)
    output_predicted(train_predict,train_out)
    output_predicted(test_predict,test_out)
    train_error = evaluate(train_data,train_predict)
    test_error = evaluate(test_data,test_predict)
    output_metrics(train_error, test_error, metrics_out)
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:11:37 2020

@author: jingxianbao
"""

import sys
import csv
import numpy as np
from scipy.sparse import csr_matrix
from feature import read_dict
import matplotlib.pyplot as plt


def create_dataset(path, word_dict):
    labels = []
    row = []
    col = []
    with open(path) as csv_file:
        csv_reader = csv_file.readlines()
        i = 0
        for line in csv_reader:
            line = line.split("\t")
            labels.append(int(line[0]))
            for feature in line[1:]:
                row.append(i)
                col.append(feature.split(":")[0])
            i += 1
    y = np.array(labels)
    x = csr_matrix((np.ones(len(row)), (row, col)), shape = (len(labels), len(word_dict.keys())))
    return {'X':x, 'y':y}


def sigmoid(u):
    return 1/(1+np.exp(-u))


def train(train_data, val_data, epoch, learning_rate = 0.1):
    train_x = train_data['X']
    train_y = train_data['y']
    val_x = val_data['X']
    val_y = val_data['y']
    train_losses = []
    val_losses = []
    w = np.zeros(train_x.shape[1])
    b = 0
    for k in range(epoch):
        print ("Epoch " + str(k))
        for i in range(len(train_y)):
            x = train_x[i].toarray()[0]
            y = train_y[i]
            grad_w = - (y - sigmoid(np.dot(x,w)+b)) * x
            grad_b = - (y - sigmoid(np.dot(x,w)+b))
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b

        train_loss = 0
        val_loss = 0
        for i in range(len(train_y)):
            x = train_x[i].toarray()[0]
            y = train_y[i]
            train_loss += -y*(np.dot(x,w)+b)+np.log(1+np.exp(np.dot(x,w)+b))
        train_losses.append(train_loss/len(train_y))

        val_loss = 0
        for i in range(len(val_y)):
            x = val_x[i].toarray()[0]
            y = val_y[i]
            val_loss += -y*(np.dot(x,w)+b)+np.log(1+np.exp(np.dot(x,w)+b))
        val_losses.append(val_loss/len(val_y))
    
    plt.plot(list(range(epoch)), train_losses,label="train")
    plt.plot(list(range(epoch)), val_losses,label="val")
    plt.xlabel('num of epoch')
    plt.ylabel('negative log likelihood ')
    plt.legend()
    plt.savefig('model2.png')
    return (w,b)


def predict(model, data_x):
    predicted_y = []
    for i in range(data_x.shape[0]):
        x = data_x[i].toarray()[0]
        p = sigmoid(np.dot(x,model[0])+model[1])
        if p >= 0.5:
            predicted_y.append(1)
        else:
            predicted_y.append(0)
    return predicted_y


def evaluate(data, predicted):
    total = len(predicted)
    error = 0
    for i in range(total):
        if data[i] != predicted[i]:
            error += 1
    return error/total
    
def output_metrics(train_err, test_err, out_file):
    print (train_err)
    print (test_err)
    with open(out_file, "w") as output:
        output.write("error(train): {0:.6f}\n".format(train_err))
        output.write("error(test): {0:.6f}".format(test_err))

def output_predicted(predicted, out_file):
    with open(out_file, "w") as output:
        for result in predicted:
            output.write(str(result)+'\n')
            
        
if __name__ == '__main__':
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    epoch = int(sys.argv[8])
    
    word_dict = read_dict(dict_in)
    
    train_data = create_dataset(train_in, word_dict)
    val_data = create_dataset(val_in, word_dict)
    test_data = create_dataset(test_in, word_dict)
    
    model = train(train_data, val_data, epoch)
    train_predict = predict(model, train_data['X'])
    test_predict = predict(model, test_data['X'])
    output_predicted(train_predict,train_out)
    output_predicted(test_predict,test_out)
    
    train_error = evaluate(train_data["y"],train_predict)
    test_error = evaluate(test_data["y"],test_predict)
    output_metrics(train_error, test_error, metrics_out)
    
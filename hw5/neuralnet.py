#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:28:10 2020

@author: jingxianbao
"""

import sys
import numpy as np

def create_dataset(path):
    labels = []
    features = []
    with open(path) as csv_file:
        csv_reader = csv_file.readlines()
        for line in csv_reader:
            line = line.split(",")
            labels.append(int(line[0]))
            features.append([float(k) for k in line[1:]])
    rows = np.arange(len(labels))
    one_hot = np.zeros((len(labels),num_classes))
    one_hot[rows, labels] = 1
    x = np.array(features)
    return {'X':x, 'y':one_hot, 'labels':labels}


class Linear():
    def __init__(self, W):
        self.W = W
        self.dW = np.zeros((W.shape[0], W.shape[1]))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = np.insert(x, 0, 1, axis = 0)
        self.out = np.dot(self.W, self.x)
        #print ('linear out')
        #print (self.out)
        return self.out

    def backward(self, delta):
        self.dW = np.dot(delta, self.x.T)
        return np.dot(self.W[:,1:].T,delta)



class Sigmoid():
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        self.state = 1/(1 + np.exp(-x))
        return self.state

    def derivative(self):
        return self.state * (1-self.state)
    

class Softmax():
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        self.logits = x
        x_exp = np.exp(x)
        self.probs = x_exp/np.sum(x_exp, axis = 0, keepdims = True)
        return self.probs

class CrossEntropy():
    def __call__(self, y, expected):
        return self.forward(y, expected)
    def forward(self, y, expected):
        self.y = y
        self.expected = expected
        #print ('expected')
        #print (self.expected)
        return np.sum(np.multiply(self.expected, np.log(self.y))*(-1))

    

def weight_init(a, b, c, flag):
    if flag == 1: # random
        alpha = np.random.rand(b, a+1) * 0.2 - 0.1
        beta = np.random.rand(c, b+1) * 0.2 - 0.1
    elif flag == 0:
        alpha = np.array([[1,1,2,-3,0,1,-3],
                         [1,3,1,2,1,0,2],
                         [1,2,2,2,2,2,1],
                         [1,1,0,2,1,-2,2]])
        beta = np.array([[1,1,2,-2,1],
                         [1,1,-1,1,2],
                         [1,3,1,-1,1]])
    else: #2: 0
        alpha = np.zeros((b, a+1))
        beta = np.zeros((c, b+1))
    return alpha, beta

class NN():
    def __init__(self, in_dim, hidden_units, out_dim, init_flag, lr):
        self.alpha, self.beta = weight_init(in_dim, hidden_units, out_dim, init_flag)
        self.lr = lr
        self.linear1 = Linear(self.alpha)
        self.activation1 = Sigmoid()
        self.linear2 = Linear(self.beta)
        self.layers = [self.linear1,self.activation1,self.linear2]
        self.output_label = Softmax()
        self.loss_fn = CrossEntropy()


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        self.out = self.output_label(out)
        return self.out
    
    def get_loss(self, labels):
        self.loss = self.loss_fn(self.out, labels)
        return self.loss
        
    def backward(self, labels):
        delta = self.out - labels
        delta = self.linear2.backward(delta)
        delta = delta * self.activation1.derivative()
        delta = self.linear1.backward(delta)
    
    def step(self):
        self.linear1.W = self.linear1.W - self.lr * self.linear1.dW
        self.alpha = self.linear1.W
        self.linear2.W = self.linear2.W - self.lr * self.linear2.dW
        self.beta = self.linear2.W
        #print ('alpha')
        #print(self.alpha)
        #print ('beta')
        #print(self.beta)
    
def train(model, trainset, testset, nepochs, lr):
    trainx = trainset['X']
    trainy = trainset['y']
    testx = testset['X']
    testy = testset['y']
    
    train_losses = np.zeros(nepochs)
    test_losses = np.zeros(nepochs)
        
    for e in range(nepochs):
        for i in range(0, len(trainx)):
            
            x = trainx[i].reshape(len(trainx[i]),1)
            y = trainy[i].reshape(num_classes, 1)
            
            model(x)
            model.backward(y)
            #print ('model out')
            #print (model.out)
            #print ('loss')
            model.get_loss(y)
            print (model.get_loss(y))
            model.step()
        train_out,train_losses[e] = evaluate(model, trainx, trainy)
        test_out,test_losses[e] = evaluate(model, testx, testy)
    return (train_out, train_losses, test_out,test_losses)



def evaluate(model, x, y):   
    n = x.shape[0]
    x = x.T
    y = y.T
    
    out = model.forward(x)
    loss = model.get_loss(y)
    #print (y)
    print (model.out)
    return np.argmax(out, axis = 0), loss/n
    
def output_predicted(predicted, out_file):
    with open(out_file, "w") as output:
        for result in predicted:
            output.write(str(result)+'\n')
            
def output_metrics(train_losses, test_losses, train_err, test_err, out_file):
    print (train_losses)
    print (test_losses)
    print (train_err)
    print (test_err)
    with open(out_file, "w") as output:
        for i in range(len(train_losses)):
            output.write("epoch={} crossentropy(train): {}\n".format(i+1, train_losses[i]))
            output.write("epoch={} crossentropy(test): {}\n".format(i+1, test_losses[i]))
        output.write("error(train): {0:.6f}\n".format(train_err))
        output.write("error(test): {0:.6f}".format(test_err))
        
        
def error_rate(data, predicted):
    total = len(predicted)
    error = 0
    for i in range(total):
        if data[i] != predicted[i]:
            error += 1
    return error/total


if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8]) #1:random, 2: 0
    learning_rate = float(sys.argv[9])
    num_classes = 3

    train_data = create_dataset(train_in)
    test_data = create_dataset(test_in)
    
    model = NN(train_data['X'].shape[1],hidden_units, num_classes, init_flag, learning_rate)
    print (model.alpha)
    print (model.beta)
    train_predict, train_losses, test_predict, test_losses = train(model,train_data,test_data,epoch,learning_rate)
    print (model.alpha)
    print (model.beta)
    
    output_predicted(train_predict,train_out)
    output_predicted(test_predict,test_out)
    
    train_error = error_rate(train_data["labels"],train_predict)
    test_error = error_rate(test_data["labels"],test_predict)
    output_metrics(train_losses, test_losses, train_error, test_error, metrics_out)
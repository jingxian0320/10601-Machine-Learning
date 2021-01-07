#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:42:12 2020

@author: jingxianbao
"""

from neuralnet import *
import matplotlib.pyplot as plt



train_data = create_dataset("largeTrain.csv")
test_data = create_dataset("largeTest.csv")
train_error_list = []
test_error_list = []

hidden_list = [5, 20, 50, 100, 200]
for i in hidden_list:
    print (i)
    model = NN(train_data['X'].shape[1],i, 10, 1, 0.01)
    _, train_losses, _, test_losses = train(model,train_data,test_data,100,0.01)
    train_error_list.append(train_losses[-1])
    test_error_list.append(test_losses[-1])
    print (train_error_list)
    print (test_error_list)




plt.plot(hidden_list, train_error_list, color='skyblue', label="Train Cross-Entropy")
plt.plot(hidden_list, test_error_list, color='olive',label="Test Cross-Entropy")
plt.legend()
plt.xlabel('# hidden units')
plt.ylabel('avg cross-entropy')
plt.show()

lr_list = [0.1, 0.01, 0.001]
fig, axs = plt.subplots(len(lr_list), figsize=(6, 9))
for i in range(len(lr_list)):
    print (i)
    model = NN(train_data['X'].shape[1],50, 10, 1, lr_list[i])
    _, train_losses, _, test_losses = train(model,train_data,test_data,100,lr_list[i])
    
    axs[i].plot(list(range(100)), train_losses, color='skyblue', label="Train Cross-Entropy")
    axs[i].plot(list(range(100)), test_losses, color='olive',label="Test Cross-Entropy")
    axs[i].set_title('loss for learning rate = ' + str(lr_list[i]))
    axs[i].set_xlabel('# epoch')
    axs[i].set_ylabel('avg cross-entropy')
    axs[i].legend()
       
plt.tight_layout()
plt.show()
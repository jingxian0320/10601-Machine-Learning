#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:37:06 2020

@author: jingxianbao
"""

from decisionTree import *
import matplotlib.pyplot as plt



train_data = read_csv("politicians_train.tsv")
test_data = read_csv("politicians_test.tsv")
n = len(train_data["values"][0])
train_error_list = []
test_error_list = []
for i in range(n):
    model = DecisionTree(i)
    model.fit(train_data)
    train_predict = model.predict(train_data)
    test_predict = model.predict(test_data)
    train_error_list.append(model.evaluate(train_data["values"],train_predict))
    test_error_list.append(model.evaluate(test_data["values"],test_predict))




plt.plot(list(range(n)), train_error_list, color='skyblue', label="Train Error")
plt.plot(list(range(n)), test_error_list, color='olive',label="Test Error")
plt.legend()
plt.show()
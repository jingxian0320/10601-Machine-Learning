#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:21:04 2020

@author: jingxianbao
"""
import learnhmm
import forwardbackward
import matplotlib.pyplot as plt

train_lls = []
test_lls = []

x = [10, 100, 1000, 10000]
for i in x:
    data = learnhmm.read_csv("trainwords.txt", i)
    word_dict = learnhmm.to_dict("index_to_word.txt")
    tag_dict = learnhmm.to_dict("index_to_tag.txt")
    prior, trans, emit = learnhmm.process(data, word_dict, tag_dict)
    
    train_data = forwardbackward.read_csv("trainwords.txt")
    test_data = forwardbackward.read_csv("testwords.txt")
    
    word2ind, ind2word = forwardbackward.to_dict("index_to_word.txt")
    tag2ind, ind2tag = forwardbackward.to_dict("index_to_tag.txt")
    
    train_ll, _, _ = forwardbackward.evaluate(train_data, prior, trans, emit, ind2word, ind2tag, word2ind, tag2ind)
    train_lls.append(train_ll)
    test_ll, _, _ = forwardbackward.evaluate(test_data, prior, trans, emit, ind2word, ind2tag, word2ind, tag2ind)
    test_lls.append(test_ll)
    
print (train_lls)
print (test_lls)

plt.plot([str(i) for i in x], train_lls, color='skyblue', label="Train")
plt.plot([str(i) for i in x], test_lls, color='olive',label="Test")
plt.legend()
plt.xlabel('# sequences used for training')
plt.ylabel('avg Log Likelihood')
plt.show()
    

    



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:02:01 2020

@author: jingxianbao
"""

import sys
from inspection import read_csv


class DecisionTree(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.depth = 0
        self.first_node = None

    def fit(self,data):
        outcomes = sorted(list(set(row[-1] for row in data["values"])))
        self.first_node = Node(data, outcomes, self.max_depth)
        self.first_node.search_next_node()
        return self
    
    def predict(self, data):
        data = data["values"]
        predicted = []
        for row in data:
            predicted.append(self.first_node.predict(row))
        return predicted

    def evaluate(self, data, predicted):
        total = len(predicted)
        error = 0
        for i in range(total):
            if data[i][-1] != predicted[i]:
                error += 1
        return error/total

class Node(object):
    def __init__(self, data, outcomes, max_depth = 10, parent = None):
        self.parent = parent
        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1
        self.data = data
        self.max_depth = max_depth
        self.gini, self.data_count = self.inspect()
        self.is_leaf = False
        self.split_ind = None
        self.model = None
        self.outcomes = outcomes
        print (self)
    
    def __str__(self):
        result = []
        for outcome in self.outcomes:
            if outcome in self.data_count:
                result.append(str(self.data_count[outcome]) + " " + str(outcome))
            else:
                result.append(str(0) + " " + str(outcome))
        s = "["
        s += "/".join(result)
        s += "]"
        return s
    
    def inspect(self):
        count = {}
        total_count = 0
        gini = 1
        for row in self.data["values"]:
            total_count += 1
            y = row[-1]
            if y not in count:
                count[y] = 1
            else:
                count[y] += 1
        for key,value in count.items():
            gini -= (value/total_count)**2
        return gini, count
    
    def decision_stump_fit(self, split_ind = 0):
        data = self.data["values"]
        total_count = len(data)
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
        
        gini = 0
        for split in count.values():
            split_count = sum(split.values())
            split_gini = 1
            for key, value in split.items():
                split_gini -= (value/split_count)**2
            gini += split_count/total_count*split_gini
        
        model = {}
        for split in count.keys():
            decision = sorted(list(count[split].items()),key=lambda x:(x[1],x[0]),reverse=True)[0][0]
            model[split] = decision
        return (split_ind, gini, model)


    def search_next_node(self):
        data = self.data["values"]
        columns = self.data["columns"]
        if len(set([x[-1] for x in data])) == 1: #purity
            self.is_leaf = True
            self.value = data[0][-1]
            return
        if self.depth > self.max_depth: #exceed max depth
            self.is_leaf = True
            self.value = sorted([(key, value) for key, value in self.data_count.items()], key = lambda x: (x[1],x[0]), reverse = True)[0][0]
            return
        result = []
        for split_ind in range(len(data[0])-1):
            if (self.parent is None) or (split_ind != self.parent.split_ind):
                result.append(self.decision_stump_fit(split_ind))
                
        split_ind, lowest_gini, model = sorted(result, key = lambda x:x[1])[0]
        
        if self.gini <= lowest_gini:
            self.is_leaf = True
            self.value = sorted([(key, value) for key, value in self.data_count.items()], key = lambda x: (x[1],x[0]), reverse = True)[0][0]
            return
        
        self.split_ind = split_ind
        self.model = model
        for key in self.model.keys():
            print ("| "*self.depth,end ="")
            print (columns[self.split_ind] + " = " + key + ": ",end ="")
            new_data = {"columns":columns,"values":[row for row in data if row[self.split_ind] == key]}
            child_node = Node(new_data, self.outcomes, self.max_depth, self)
            self.model[key] = child_node
            child_node.search_next_node()
            
    def predict(self, row):
        if self.is_leaf:
            return self.value
        else:
            return self.model[row[self.split_ind]].predict(row)
        
        
def output_metrics(train_err, test_err, out_file):
    print (train_err)
    print (test_err)
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
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    #print("The train input file is:%s" % (train_in))
    #print("The test input file is:%s" % (test_in))
    #print("The max depth is:%s" % (max_depth))
    #print("The train output file is:%s" % (train_out))
    #print("The test output file is:%s" % (test_out))
    #print("The metrics out file is:%s" % (metrics_out))
    
    train_data = read_csv(train_in)
    test_data = read_csv(test_in)
    
    model = DecisionTree(max_depth)
    model.fit(train_data)
    train_predict = model.predict(train_data)
    test_predict = model.predict(test_data)
    output_predicted(train_predict,train_out)
    output_predicted(test_predict,test_out)
    train_error = model.evaluate(train_data["values"],train_predict)
    test_error = model.evaluate(test_data["values"],test_predict)
    output_metrics(train_error, test_error, metrics_out)
    
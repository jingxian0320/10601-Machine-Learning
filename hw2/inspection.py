#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 00:43:38 2020

@author: jingxianbao
"""

import sys
import csv


def read_csv(path):
    #print (path)
    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
                columns = row
            else:
                data.append(row)
                line_count += 1
    #print('Processed %d lines.'%line_count)
    return {"values":data, "columns":columns}

def inspect(data):
    count = {}
    total_count = 0
    gini = 1
    for row in data:
        total_count += 1
        y = row[-1]
        if y not in count:
            count[y] = 1
        else:
            count[y] += 1
    for key,value in count.items():
        gini -= (value/total_count)**2
    error = 1 - max(count.values())/total_count
    return gini,error

def output_result(gini,error,out_file):
    with open(out_file, "w") as output:
        output.write("gini_impurity: {0:.4f}\n".format(gini))
        output.write("error: {0:.4f}".format(error))


if __name__ == '__main__':
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    print("The input file is:%s" % (file_in))
    print("The output file is:%s" % (file_out))
    data = read_csv(file_in)["values"]
    gini,error = inspect(data)
    output_result(gini,error,file_out)
    
    
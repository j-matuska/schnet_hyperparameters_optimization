#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:48:29 2023

@author: jamat
"""

import numpy
import pickle

class LL_stats:
    def __init__(self, hyperparameters = None, LL_lines = None, xx = None):
        self.hyperparameters = hyperparameters
        self.LL_lines = numpy.array(LL_lines)     
        self.x = numpy.array(xx)
        self.entropy = None
        
    def calculate_entropy(self, T, k=1.0):
        x = self.LL_lines.mean(axis = 1)
        print(x.shape)
        y = x #- x.min()
        beta = 1 / (k * T)
        Q = numpy.sum(numpy.exp(-beta * y), axis = 1)
        self.entropy = k * numpy.log(Q)
        print(self.entropy.shape)
        return self.entropy
    
    def get_entropy(self, T=1e4):
        if self.entropy is None:
            self.calculate_entropy(T)
        return self.entropy
    
    def save(self, name0):
        name = name0+".LL_stats"
        with open(name, 'w') as file:
           pickle.dump(self, file)
        return 0
    
    def load(self, name0):
        name = name0+".LL_stats"
        with open(name, 'r') as file:
            dummy_class = pickle.load(file) # zaciatok nie dobry
        return dummy_class
           
    
           

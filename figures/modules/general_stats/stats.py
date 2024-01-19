#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:51:17 2023

@author: jamat
"""

import numpy
from general_stats.calculations import calculate_classification_statistics, calculate_LSQM, calculate_MSE, calculate_DS_gaussian

# class run_stats:
#     def __init__(self, 
#                  hyperparameters, 
#                  mse, 
#                  slope, 
#                  intercept, 
#                  R2, 
#                  intervaly, 
#                  mse_ds, 
#                  rel_pocet_ds, 
#                  mu_ds, 
#                  sigma_ds, 
#                  fraction_pocet_ds, 
#                  recall,
#                  specificity,
#                  precision):
#         self.hyperparameters = hyperparameters
#         self.mse = numpy.array(mse)
#         self.slope = numpy.array(slope)
#         self.intercept = numpy.array(intercept)
#         self.R2 = numpy.array(R2)
#         self.intervaly = numpy.array(intervaly)
#         self.mse_ds = numpy.array(mse_ds)
#         self.rel_pocet_ds = numpy.array(rel_pocet_ds)
#         self.mu_ds = numpy.array(mu_ds)
#         self.sigma_ds = numpy.array(sigma_ds)
#         self.fraction_pocet_ds = numpy.array(fraction_pocet_ds)
#         self.recall =  numpy.array(recall)
#         self.specificity =  numpy.array(specificity)
#         self.precision =  numpy.array(precision)

class run_stats:
    def __init__(self, 
                 hyperparameters, 
                 name,
                 original,
                 predicted
                 ):
        self.hyperparameters = hyperparameters
        self.name = numpy.array(name)
        self.original = numpy.array(original)
        self.predicted = numpy.array(predicted)
        self.calculate_all()

    def calculate_all(self, threshold = -8.6):
        
        n_split, n_molecules = self.name.shape
        print(n_split, n_molecules)
        
        mse = []
        slope = []
        intercept = []
        R2 = []
        recall = []
        specificity = []
        precision = []

        intervaly = []
        mse_ds = []
        rel_pocet_ds = []
        mu_ds = []
        sigma_ds = []
        fraction_over_2_ds = []
        
        recall_PRC = []
        precision_PRC = []
        
        for n,o,p in zip(self.name,self.original,self.predicted):
            
            mse.append(calculate_MSE(o, p))
            slope0, intercept0, R20 = calculate_LSQM(o, p)
            
            slope.append(slope0)
            intercept.append(intercept0)
            R2.append(R20)
            
            recall0, specificity0, precision0 = calculate_classification_statistics(o, p, threshold)
            
            recall.append(recall0)
            specificity.append(specificity0)
            precision.append(precision0)
            
            intervaly0, mse_ds0, rel_pocet_ds0, mu_ds0, sigma_ds0, fraction_over_2_ds0 = calculate_DS_gaussian(o, p)
            
            intervaly.append(intervaly0)
            mse_ds.append(mse_ds0)
            rel_pocet_ds.append(rel_pocet_ds0)
            mu_ds.append(mu_ds0)
            sigma_ds.append(sigma_ds0)
            fraction_over_2_ds.append(fraction_over_2_ds0)
            
            #PRC_recall, PRC_precision, baseline =  calculate_PRC(o, p, threshold)
            
            #recall_PRC.append(PRC_recall)
            #precision_PRC.append(PRC_precision)
            
        self.mse = numpy.array(mse)
        self.slope = numpy.array(slope)
        self.intercept = numpy.array(intercept)
        self.R2 = numpy.array(R2)
        self.intervaly = numpy.array(intervaly)
        self.mse_ds = numpy.array(mse_ds)
        self.rel_pocet_ds = numpy.array(rel_pocet_ds)
        self.mu_ds = numpy.array(mu_ds)
        self.sigma_ds = numpy.array(sigma_ds)
        self.fraction_pocet_ds = numpy.array(fraction_over_2_ds)
        self.recall =  numpy.array(recall)
        self.specificity =  numpy.array(specificity)
        self.precision =  numpy.array(precision)
        self.recall_thr_dependence = numpy.array(recall_PRC)
        self.precision_thr_dependence = numpy.array(precision_PRC)
        #self.baseline = float(baseline)

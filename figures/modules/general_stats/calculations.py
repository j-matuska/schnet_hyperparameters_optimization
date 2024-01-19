#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:30:40 2023

@author: jamat
"""

import numpy
from scipy import stats
from sklearn.metrics import confusion_matrix


def calculate_LSQM(x: numpy.dtype(numpy.float64), y: numpy.dtype(numpy.float64)):
    """
    Calculate parameters of least square method

    Parameters
    ----------
    x : numpy.dtype(numpy.float64)
        DESCRIPTION.
    y : numpy.dtype(numpy.float64)
        DESCRIPTION.

    Returns
    -------
    slope : float
        Slope of the line.
    intercept : float
        Intercept of the line.
    R2 : float
        Correlation coefficient.

    """
    
    result = stats.linregress(x.astype(float), y.astype(float))

    slope = float(result.slope)
    intercept = float(result.intercept)
    R = float(result.rvalue)
    R2 = R*R
    
    return slope, intercept, R2


def calculate_DS_gaussian(original, predicted):   
    
    AD_min=-15
    AD_max=1
    intervaly = numpy.linspace(AD_min, AD_max, 9)
    
    error = numpy.subtract(original,predicted)
    # MAE
    ae = numpy.absolute(error)
    
    sigma_ds = numpy.zeros(intervaly.size-1,dtype=numpy.float_)
    mu_ds = numpy.zeros(intervaly.size-1,dtype=numpy.float_)
    mse_ds = numpy.zeros(intervaly.size-1,dtype=numpy.float_)
    pocet_ds = numpy.zeros(intervaly.size-1,dtype=numpy.int_)
    fraction_pocet_ds = numpy.zeros(intervaly.size-1,dtype=numpy.float_)
    rel_pocet_ds = numpy.zeros(intervaly.size-1,dtype=numpy.float_)
    
    for i in numpy.arange(intervaly.size-1):
        
        umiestnenie_bool = numpy.logical_and(original > intervaly[i], original <= intervaly[i+1])
        if i == 0: # ak menej ako najmenej, tak priradit do najmensej kategorie
            doplnok = (original < intervaly[i])
            umiestnenie_bool[doplnok] = True
        if i == intervaly.size-2: #ak viacej ako najviac, tak priradit do najvacsej kategorie
            doplnok = (original > intervaly[i+1])
            umiestnenie_bool[doplnok] = True
        umiestnenie = numpy.where(umiestnenie_bool)
        rel_pocet_ds[i] = umiestnenie[0].size/umiestnenie_bool.size    
        sigma_ds[i] = numpy.std(error[umiestnenie],ddof=0) #zmenil som ddof 1 na 0, lebo to nefungovalo ak tam ostala iba jedna polozka
        mu_ds[i] = numpy.mean(error[umiestnenie])
        mse_ds[i] = numpy.square(error[umiestnenie]).mean()
        pocet_ds[i] = numpy.count_nonzero(ae[umiestnenie] > 2.0)
        fraction_pocet_ds[i] = pocet_ds[i]/umiestnenie[0].size
 
    return intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_pocet_ds


def calculate_MSE(original,predicted):
    """
    Calculation of MSE from set.
    
    Parameters
    ----------
    original: array
        Original data
    predicted: array
        Predicted data

    Returns
    -------
    mse : float
        Mean square error.

    """
    remove_outliers = numpy.abs(numpy.subtract(original,predicted)) < 200 
    mse = numpy.square(numpy.subtract(original,predicted)[remove_outliers]).mean()
    return mse


def calculate_classification_statistics(original: numpy.dtype(numpy.float_), predicted: numpy.dtype(numpy.float_), threshold: float):
    
    thr_original = numpy.array(original < threshold)
    thr_predicted = numpy.array(predicted < threshold)
    
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(thr_original, thr_predicted).ravel()

    true_positive_rate = true_positive/(true_positive+false_negative)
    false_positive_rate = false_positive/(true_negative+false_positive)
    positive_predictive_value = true_positive/(true_positive+false_positive)
    
    recall = true_positive_rate 
    specificity = 1-false_positive_rate
    precision = positive_predictive_value
    
    return recall, specificity, precision

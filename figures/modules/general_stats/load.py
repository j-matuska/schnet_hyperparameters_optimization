#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:32:22 2023

@author: jamat
"""

import pandas
import numpy
from general_stats.stats import run_stats

def read_csv(name_csv):
    """
    Read csv file, extract name, original and predicted score of the molecule.

    Parameters
    ----------
    filename : str
        Filename of the csv file.

    Returns
    -------
    name : array of strings
        Array containing names of the molecules.
    original : array of floats
        Array containing original scores.
    predicted : array of floats
        Array containing original scores.

    """
    
    # Nacitanie a transformovanie dat
    df = pandas.read_csv(name_csv, sep=';')
    data = df.values.transpose()

    #print(df.columns[0])

    # Vyberieme stlpce z dat z ktorych sa bude robit graf
    name = data[0] # coumpounds labels
    original = data[1]
    predicted = data[2]
    
    return name, original.astype(float), predicted.astype(float) 

def load_csv_files(name0: str, runs: list[dict[str,str]], split: numpy.dtype(numpy.int64)):
    
    output = []
    
    for i,r in enumerate(runs):
        
        n = []
        o = []
        p = []
        
        for l in split:
            name_csv = "data/{s}/0{l}/{name0}_0{l}.csv".format(name0 = name0,
                                                               s = r.get("adresar"),
                                                               l = l
                                                               )
            name, original, predicted = read_csv(name_csv)
            n.append(name)
            o.append(original)
            p.append(predicted)
     
        output.append(
            run_stats(
                r,
                numpy.array(n),
                numpy.array(o),
                numpy.array(p)
                )
            )
            
    return output

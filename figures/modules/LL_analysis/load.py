#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:53:10 2023

@author: jamat
"""

import numpy
from LL_analysis.LL_stats import LL_stats


def load_npy_files(runs: list[dict[str,str]], split: numpy.dtype(numpy.int64), distance: float):
    
    output = []

    for i,r in enumerate(runs):
        
        LL_lines = []
        xx = []
        
        for l in split:
            name_npy = "data/{s}/0{l}/ip_explorer/{name0}.npy".format(name0 = r.get("loss_landscape_file"), 
                                                                                    s = r.get("adresar"),
                                                                                    l = l
                                                                                    )
            x, data = read_npy(name_npy, r.get("distance"))
            LL_lines.append(data)
            xx.append(x)
            
        output.append(
            LL_stats(
                r, 
                LL_lines,
                xx
                )
            )
            
    return output

def read_npy(name_npy: str, distance: float):
    
    data = numpy.load(name_npy)
    print(data.shape)
    n,steps = data.shape
    
    x = numpy.atleast_2d(numpy.linspace(0, distance, steps)).repeat(n, axis = 0)
    
    return x[:,:21],data[:,:21]


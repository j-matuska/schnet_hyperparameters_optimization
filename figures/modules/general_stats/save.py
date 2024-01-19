#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:37:41 2023

@author: jamat
"""

import numpy


def save_classification_statistics(name0: str, xdata: list[numpy.dtype(numpy.float_)], mse: list[numpy.dtype(numpy.float_)], labels: list[str], x_label: str, y_label: str):
    
    name = "{}_CS.txt".format(name0)
    
    with open(name,'w') as file:
        file. write("{}; {}; {}\n". format(x_label, y_label, "label"))
        
        for x,y,l in zip(xdata,mse,labels):
            print(x,y,l)
            zoradene = numpy.argsort(x)
            
            for a,b in zip(x[zoradene],y[zoradene,:]):
                line = "{}; {:.3f} +/- {:.3f}; {}\n".format(
                    a, 
                    numpy.mean(b,axis=-1), 
                    numpy.std(b,axis=-1,ddof=0), 
                    l
                    )
                file.write(line)
       
    return 0

def save_table(name0: str, runs: list[dict[str,str]], data: numpy.dtype(numpy.float64), label: str):
    
    label0 = label.split(" ")[0]
    
    name = "{}_{}_3.txt".format(name0, label0)
    
    data_m = data.mean(axis=-1)
    data_std = data.std(axis=-1, ddof=0)
    
    with open(name,'w') as file:
        file. write("representation; batch size; number of parameters/k; {}\n". format(label))
        
        for i,r in enumerate(runs):
            line = "{representation}; {f}; {s}; {d:.3f} +/- {std:.3f}\n".format(representation=r.get("representation"),
                                                        f = r.get("split_suffix"),
                                                        s = r.get("adresar"),
                                                        d = data_m[i],
                                                        std = data_std[i]
                                                        )
            file.write(line)
        
    return 0 
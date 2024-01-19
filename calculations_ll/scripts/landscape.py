#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:38:30 2023

@author: j-matuska

Helper script to concrecate lines files from landspace*.sh
Workaround around error in ip_explorer.landscape

"""

import numpy

def main():
    
    raw_data = []
    
    for i in range(20):
        data = numpy.load(f'{i}lines=DS_d=0.20_s=25_schnetDS.npy')
        # trik, lebo som divne tvoril tie subory
        if len(data.shape) > 1:
            raw_data.append(data[0,:])
        else:
            raw_data.append(data)
        
    data_array = numpy.array(raw_data)
    
    print(data_array)
    print(data_array.shape)
    a,b=data_array.shape
    print(a,b)
    
    numpy.save(f'lines=DS_d=0.20_s=25_{a}_schnetDS.npy', data_array)
    
    return 0

if __name__ == '__main__':
    main()

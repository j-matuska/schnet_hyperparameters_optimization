#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:00:08 2023

@author: j-matuska
"""

import sys
import os
import numpy
from ase.io import read
from DS_histogram import plot_histogram

def main():
    
    xyz_files = []
    datasets = []
    
    for file_name in sys.argv[1:]:
        
        xyz_files.append(read(file_name, index=':'))
        datasets.append(os.path.splitext(os.path.split(file_name)[-1])[-2])
    
    print(datasets)
    datasets = ["in-vivo", "in-vitro-only"]
    print(datasets)
    
    name = []
    DS = []
    
    for atoms in xyz_files:
        
        name.append(numpy.array([at.info.get("name") for at in atoms]))
        DS.append(numpy.array([at.info.get("energy") for at in atoms]))
    
    AD_min=-15
    AD_max=1
    intervaly = numpy.linspace(AD_min, AD_max, 9)
    pocetnost = []
    pocetnost_BA = []
    hranice = []
    
    for DS1 in DS:
        
        p,h=numpy.histogram(numpy.where(DS1<intervaly[0],intervaly[0],DS1),bins=intervaly, density = True)
        pocetnost_BA.append(p)
        pocetnost.append(numpy.log10(p/1e-5))
        hranice.append(h)
        
    print(hranice,pocetnost)

    plot_histogram(hranice, pocetnost_BA, r'$DS_{expected}$ [kcal/mol]', "rel. abundance", labels=datasets)

    return 0

if __name__ == '__main__':
    main()
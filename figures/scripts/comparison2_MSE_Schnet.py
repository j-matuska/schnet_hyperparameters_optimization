#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:11:48 2023

@author: j-matuska
"""

import matplotlib
import json
import numpy

from general_stats.load import load_csv_files
from general_stats.plot import plot_MSE, plot_classification_statistics, plot_correlaton_densityM


def main():
    """
    It will scan predefined folder structure and gather information on:
        MSE
        slope
        intercept
        correlation coefficient
        .
        .

    Returns
    -------
    int
        0 correct exit.
        1 

    """
    
    params = {'legend.fontsize': 10,
          'figure.figsize': (3.3, 2.5),
          'figure.dpi': 300,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize':10,
         'ytick.labelsize':10,
         'text.usetex': False}
    matplotlib.pyplot.rcParams.update(params)
    
    with open("selected_runs.json","r") as r:
       selected_runs = json.load(r)
    
    split = numpy.array([1,2,3,4,5])
    
    name0 = "in_vitro"
    
    vitro_stats = load_csv_files(name0, selected_runs, split)
    
    for s in vitro_stats:
        print(s.mse_ds[:,0])
    
    x = []
    y = []
    y2 = []
    y3 = []
    y4 = []
    recalls = []
    labels = []
    
    for c in [5.0,7.5,10.0]:
        x.append(numpy.array([r.get("n_rbf") for r in selected_runs if r.get("cutoff") == c] ))
        y.append(numpy.array([s.mse for s in vitro_stats if s.hyperparameters.get("cutoff") == c]))
        y2.append(numpy.array([s.mse_ds[:,0] for s in vitro_stats if s.hyperparameters.get("cutoff") == c]))
        y3.append(numpy.array([s.mse_ds[:,1] for s in vitro_stats if s.hyperparameters.get("cutoff") == c]))
        y4.append(numpy.array([s.mse_ds[:,2] for s in vitro_stats if s.hyperparameters.get("cutoff") == c]))
        recalls.append(numpy.array([s.recall for s in vitro_stats if s.hyperparameters.get("cutoff") == c]))
        labels.append("cutoff = "+str(c)+r' $\AA$')
    print(x,y,labels)
    
    
    plot_MSE(name0, x, y, labels, r'$n_{rbf}$', r'MSE [($\mathrm{kcal/mol)^2}$]')
    
    plot_MSE(name0+"max", x, y2, numpy.zeros_like(labels), r'$n_{rbf}$', r'$\mathrm{MSE_{( -\infty;-13 \rangle }}$ [$\mathrm{(kcal/mol)^2}$]')
    plot_MSE(name0+"max2", x, y3, numpy.zeros_like(labels), r'$n_{rbf}$', r'$\mathrm{MSE_{(-13;-11\rangle}}$ [$\mathrm{(kcal/mol)^2}$]')
    plot_MSE(name0+"max3", x, y4, numpy.zeros_like(labels), r'$n_{rbf}$', r'$\mathrm{MSE_{(-11;-9\rangle}}$ [$\mathrm{(kcal/mol)^2}$]')
    
    
    print(recalls)
    plot_classification_statistics(name0+"recall", x, recalls, labels,"$n_{rbf}$", "recall")
     
    recalls = [numpy.array(s.recall) for s in vitro_stats]
    print([s.mse.shape for s in vitro_stats])
    y2 = [numpy.sqrt(s.mse_ds[:,0]) for s in vitro_stats]
    y3 = [numpy.sqrt(s.mse_ds[:,1]) for s in vitro_stats]
    y4 = [numpy.sqrt(s.mse_ds[:,2]) for s in vitro_stats]
    
    labels = [r'{:>2.1f} $\AA$; {:>2d}'.format(s.hyperparameters.get("cutoff"),s.hyperparameters.get("n_rbf")) for s in vitro_stats]

    for s in [vitro_stats[0],vitro_stats[-1]]:
        name0 = "{r}_{c}Ang".format(r = s.hyperparameters.get("representation"), c = s.hyperparameters.get("cutoff"))
        plot_correlaton_densityM(name0, s.original[0,:], s.predicted[0,:], r'$\mathrm{DS_{expected}}$ [kcal/mol]', r'$\mathrm{DS_{predicted}}$ [kcal/mol]')

    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)
    
    return 0

if __name__ == '__main__':
    main()

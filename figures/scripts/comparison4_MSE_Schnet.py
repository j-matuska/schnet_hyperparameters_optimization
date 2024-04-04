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

    json_file = "selected_runs_exp2.json"
    
    with open(json_file,"r") as r:
       selected_runs = json.load(r)
    
    split = numpy.array([1,2,3,4,5])
    
    name0 = "in_vitro"
    
    vitro_stats = load_csv_files(name0, selected_runs, split)
    
    x = []
    y = []
    y2 = []
    y3 = []
    y4 = []
    recalls = []
    labels = []
    
    for c,rbf in zip([5.0,10.0],[20,50]):
        x.append(numpy.array([r.get("n_parameters") for r in selected_runs if r.get("cutoff") == c and r.get("n_rbf") == rbf] ))
        y.append(numpy.array([s.mse for s in vitro_stats if s.hyperparameters.get("cutoff") == c and s.hyperparameters.get("n_rbf") == rbf]))
        y2.append(numpy.array([s.mse_ds[:,0] for s in vitro_stats if s.hyperparameters.get("cutoff") == c and s.hyperparameters.get("n_rbf") == rbf]))
        y3.append(numpy.array([s.mse_ds[:,1] for s in vitro_stats if s.hyperparameters.get("cutoff") == c and s.hyperparameters.get("n_rbf") == rbf]))
        y4.append(numpy.array([s.mse_ds[:,2] for s in vitro_stats if s.hyperparameters.get("cutoff") == c and s.hyperparameters.get("n_rbf") == rbf]))
        recalls.append(numpy.array([s.recall for s in vitro_stats if s.hyperparameters.get("cutoff") == c and s.hyperparameters.get("n_rbf") == rbf]))
        labels.append("cutoff = "+str(c)+r' $\AA$')
    
    
    plot_MSE(name0+"_parameters_", x, y, labels, r'$n_{parameters}/k$', r'MSE [($\mathrm{kcal/mol)^2}$]')
    
    plot_MSE(name0+"_parameters_"+"max", x, y2, numpy.zeros_like(labels), r'$n_{weights}$/k', r'$\mathrm{MSE_{( -\infty;-13 \rangle }}$ [$\mathrm{(kcal/mol)^2}$]')
    plot_MSE(name0+"_parameters_"+"max2", x, y3, numpy.zeros_like(labels), r'$n_{weights}$/k', r'$\mathrm{MSE_{(-13;-11\rangle}}$ [$\mathrm{(kcal/mol)^2}$]')
    plot_MSE(name0+"_parameters_"+"max3", x, y4, numpy.zeros_like(labels), r'$n_{weights}$/k', r'$\mathrm{MSE_{(-11;-9\rangle}}$ [$\mathrm{(kcal/mol)^2}$]')
    
    plot_classification_statistics(name0+"_parameters_recall", x, recalls, labels,"$n_{parameters}$", "recall")
    
    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)
    
    return 0

if __name__ == '__main__':
    main()

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
from LL_analysis.load import load_npy_files
from LL_analysis.plot import plot_correlaton, plot_correlaton_all2, plot_intervaly
from LL_analysis.calculations import calculate_LSQM2

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

    """
    
    params = {'legend.fontsize': 10,
          'figure.figsize': (3.3, 2.5),
          'figure.dpi': 300,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
    matplotlib.pyplot.rcParams.update(params)
    
    with open("selected_runs.json","r") as r:
        selected_runs = json.load(r)
    
    split = numpy.array([1,2,3,4,5])
    
    name0 = "in_vitro"
    
    vitro_stats = load_csv_files(name0, selected_runs, split)
    
    rmse = [numpy.sqrt(s.mse) for s in vitro_stats]
    print([s.mse.shape for s in vitro_stats])

    name0 = "in_vitro_MSE_vs_in_vivo_80_entropy"

    ll_stats = load_npy_files(selected_runs, split)
        
    for T in [1e4]:
        
        entropy = [s.calculate_entropy(T) for s in ll_stats]
        
        labels = [r'{:>2.1f} $\AA$; {:>2d}'.format(s.hyperparameters.get("cutoff"),s.hyperparameters.get("n_rbf")) for s in ll_stats]
        plot_correlaton(name0+"_corr0", numpy.array(entropy), numpy.array(rmse), "entropy", "$RMSE$[kcal/mol]", labels=labels)
    
    name0 = "in_vitro_recall_vs_in_vivo_80_entropy"
    
    recalls = [numpy.array(s.recall) for s in vitro_stats]

    for T in [1e4]:
        
        entropy= [s.calculate_entropy(T) for s in ll_stats]
        
        labels = ["{:>2.1f} Ang; {:>2d}".format(s.hyperparameters.get("cutoff"),s.hyperparameters.get("n_rbf")) for s in ll_stats]
        
        plot_correlaton(name0+"_corr0", numpy.array(entropy), numpy.array(recalls), "entropy", "recall", labels=labels)
    
    
    intervaly = [s.intervaly for s in vitro_stats]
    intervaly_a = numpy.array(intervaly)
    a,b,c = intervaly_a.shape
    print(a, b, c)
    mse_ds = [numpy.sqrt(s.mse_ds) for s in vitro_stats]
    y_a = numpy.array(mse_ds)
    a,b,c = y_a.shape
    print(a, b, c)
    entropy = numpy.array([s.calculate_entropy(T) for s in ll_stats])
    print(entropy.shape)
    
    slope = numpy.zeros(c)
    intercept = numpy.zeros(c)
    R2 = numpy.zeros(c)
    slope_std = numpy.zeros(c)

    for i in range(c):
        slope[i], intercept[i], R2[i], slope_std[i] = calculate_LSQM2(numpy.ravel(entropy), numpy.ravel(y_a[:,:,i]))
    
    print(slope, intercept, R2)
    
    plot_intervaly("Schnet_cutoff_rbf", intervaly_a[0,0,:], slope, slope_std, r'$\mathrm{DS_{expected}}$ [kcal/mol]', "slope [RMSE/entropy]")
    
    name0 = "in_vitro_MSE_vs_in_vivo_80_entropy"
    
    #labels2 = [r'(-$\infty$;-13$\rangle$ kcal/mol', r'(-13;-11$\rangle$ kcal/mol', r'(-11;-9$\rangle$ kcal/mol' ]
    labels2 = [r'(-$\infty$;-13$\rangle$', r'(-13;-11$\rangle$', r'(-11;-9$\rangle$' ]

    plot_correlaton_all2(
        name0+"_corrX", 
        numpy.array(entropy[:,:]), 
        numpy.array(y_a[:,:,:3]), 
        numpy.array(rmse), 
        "entropy", 
        "RMSE [kcal/mol]",
        labels = labels,
        labels2 = labels2
        )

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)
    
    return 0

if __name__ == '__main__':
    main()

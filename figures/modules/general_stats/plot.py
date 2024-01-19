#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:52:28 2023

@author: jamat
"""

import numpy
import matplotlib

from general_stats.calculations import calculate_LSQM

def plot_MSE(name0: str, xdata: list[numpy.dtype(numpy.float_)], mse: list[numpy.dtype(numpy.float_)], labels: list[str], x_label: str, y_label: str):
    """
    

    Parameters
    ----------
    name0 : str
        DESCRIPTION.
    n_parameters : list[list[int]]
        DESCRIPTION.
    representations : list[str]
        DESCRIPTION.
    size : list[list[str]]
        DESCRIPTION.
    suffix : list[list[dict[str,str]]]
        DESCRIPTION.
    mse : numpy.dtype(numpy.float_)
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    for x,y,l in zip(xdata,mse,labels):
        #print(x,y,l)
        zoradene = numpy.argsort(x)
        matplotlib.pyplot.errorbar(x[zoradene], numpy.mean(y[zoradene,:],axis=-1), yerr = numpy.std(y[zoradene,:],axis=-1,ddof=0), label = l)
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    ymin,ymax=osi.get_ylim()
    osi.set_ylim(0,ymax)
    print(labels)
    if labels[0] != '':
        matplotlib.pyplot.legend(loc="lower left")
    
    matplotlib.pyplot.tight_layout()
    name = "{}_mse_3.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    name = "{}_mse_3.eps".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    matplotlib.pyplot.close()
    
    return 0

def plot_classification_statistics(name0: str, xdata: list[numpy.dtype(numpy.float_)], mse: list[numpy.dtype(numpy.float_)], labels: list[str], x_label: str, y_label: str):

    for x,y,l in zip(xdata,mse,labels):
        print(x,y,l)
        zoradene = numpy.argsort(x)
        matplotlib.pyplot.errorbar(x[zoradene], numpy.mean(y[zoradene,:],axis=-1), yerr = numpy.std(y[zoradene,:],axis=-1,ddof=0), label = l)
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    osi.set_ylim(0,1)
    # matplotlib.pyplot.legend(loc="lower left")
    matplotlib.pyplot.tight_layout()
    name = "{}_CS_3.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    name = "{}_CS_3.eps".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    matplotlib.pyplot.close()
    
    return 0

def plot_correlaton(name0: str, xdata: numpy.dtype(numpy.float_), mse: numpy.dtype(numpy.float_), x_label: str, y_label: str, labels:list[str]):

    matplotlib.pyplot.figure()
    for x,y,l in zip(xdata,mse,labels):
        matplotlib.pyplot.plot(x, y, "o", label = l)
    
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    
    xmin,xmax = osi.get_xlim()
    ymin,ymax = osi.get_ylim()
    
    
    slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata), numpy.ravel(mse))
    matplotlib.pyplot.axline((0,intercept), slope=slope, color = "k")
    values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    osi.set_xlim(xmin,xmax)
    osi.set_ylim(ymin,ymax)
    
    osi.text(1.05, 0.05, values,transform=osi.transAxes)
    
    matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1), bbox_transform=osi.transAxes, alignment='right' )
    matplotlib.pyplot.tight_layout()
    # name = "{}.png".format(name0)
    # matplotlib.pyplot.savefig(name, dpi= 300)
    # matplotlib.pyplot.close()
    matplotlib.pyplot.show()
    
    return 0

def plot_correlaton_density( name: str, xdata: numpy.dtype(numpy.float_), ydata: numpy.dtype(numpy.float_), x_label: str, y_label: str):
    
    matplotlib.pyplot.figure()
    
    # Rozmery osi: min a max
    x_min = -17
    x_max = 1.0
    
    quantity, x_edges, y_edges = numpy.histogram2d(numpy.ravel(xdata), numpy.ravel(ydata), bins=(numpy.arange(x_min,x_max+0.1,0.1),numpy.arange(x_min,x_max+0.1,0.1)))
    
    print(quantity, x_edges, y_edges)
    
    log_quantity = numpy.log10(quantity.T)
    
    matplotlib.pyplot.imshow(log_quantity, interpolation='antialiased', origin='lower', extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]))
    
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    
    # xmin,xmax = osi.get_xlim()
    # ymin,ymax = osi.get_ylim()
    
    slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata), numpy.ravel(ydata))
    matplotlib.pyplot.axline((0,intercept), slope=slope, color = "r")
    values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    matplotlib.pyplot.axline((0,0), slope=1.0, color = "k", linestyle = "--")
    
    osi.set_xlim(x_min,x_max)
    osi.set_ylim(x_min,x_max)
    
    osi.set_xticks(numpy.linspace(x_min, x_max, 5))
    osi.set_yticks(numpy.linspace(x_min, x_max, 5))
    
    osi.text(1.05, 0.05, values, transform=osi.transAxes, color = "r")
    
    #matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1), bbox_transform=osi.transAxes, alignment='right' )
    matplotlib.pyplot.colorbar(shrink = 0.7, anchor = (0.0, 0.8), label = "thousand's")
    matplotlib.pyplot.tight_layout()
    name = "{}_correlation_density.png".format(name)
    matplotlib.pyplot.savefig(name, dpi= 300)
    matplotlib.pyplot.close()
    matplotlib.pyplot.show()
    
    return 0

def plot_correlaton_densityM( name0: str, xdata: numpy.dtype(numpy.float_), ydata: numpy.dtype(numpy.float_), x_label: str, y_label: str):
    
    matplotlib.pyplot.figure()
    
    # Rozmery osi: min a max
    x_min = -17
    x_max = 1.0
    
    quantity, x_edges, y_edges = numpy.histogram2d(numpy.ravel(xdata), numpy.ravel(ydata), bins=(numpy.arange(x_min,x_max+0.1,0.1),numpy.arange(x_min,x_max+0.1,0.1)))
    
    print(quantity, x_edges, y_edges)
    
    
    # Získanie indexu binu pre každú hodnotu v "expected" a "predicted"
    x_indices = numpy.digitize(xdata, x_edges)
    y_indices = numpy.digitize(ydata, y_edges)

    # Korekcia indexov, ktoré sú mimo rozsahu
    x_indices[x_indices > len(x_edges) - 1] = len(x_edges) - 1
    y_indices[y_indices > len(y_edges) - 1] = len(y_edges) - 1

    # Získanie početnosti pre každý bod podľa jeho indexu binu
    counts_2d = quantity[x_indices - 1, y_indices - 1]
    
    # Normalizácia početnosti na rozsah 0-1
    norm_counts_2d = counts_2d / numpy.max(quantity)
    log_quantity = numpy.log10(counts_2d)
    
    matplotlib.pyplot.scatter(
        numpy.ravel(xdata), numpy.ravel(ydata), 
        c = log_quantity, s = 5,
        cmap=matplotlib.pyplot.cm.viridis
        )
    
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    
    # xmin,xmax = osi.get_xlim()
    # ymin,ymax = osi.get_ylim()
    
    slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata), numpy.ravel(ydata))
    matplotlib.pyplot.axline((0,intercept), slope=slope, color = "r")
    values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    matplotlib.pyplot.axline((0,0), slope=1.0, color = "k", linestyle = "--")
    
    osi.set_xlim(x_min,x_max)
    osi.set_ylim(x_min,x_max)
    
    osi.set_xticks(numpy.linspace(x_min, x_max, 5))
    osi.set_xticklabels(numpy.linspace(x_min, x_max, 5), fontsize = 8)
    osi.set_yticks(numpy.linspace(x_min, x_max, 5))
    osi.set_yticklabels(numpy.linspace(x_min, x_max, 5), fontsize = 8)
    
    osi.text(1.05, 0.05, values, transform=osi.transAxes, color = "r", fontsize = 8)
    
    #matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1), bbox_transform=osi.transAxes, alignment='right' )
    matplotlib.pyplot.colorbar(shrink = 0.7, anchor = (0.0, 0.7), label = "$10^n$")
    matplotlib.pyplot.tight_layout()
    name = "{}_correlation_densityM.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    name = "{}_correlation_densityM.eps".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    matplotlib.pyplot.close()
    # matplotlib.pyplot.show()
    
    return 0


def plot_precision_recall(recall_all, precision_all, labels):
    
    matplotlib.pyplot.figure("precision_recall")
    
    osi = matplotlib.pyplot.gca()
    
    for recall, precision, label in zip(recall_all, precision_all, labels):
        matplotlib.pyplot.errorbar(
            numpy.mean(recall, axis=-1), 
            numpy.mean(precision, axis=-1), 
            xerr=numpy.std(recall, axis=-1, ddof=0), 
            yerr=numpy.std(precision, axis=-1, ddof=0), 
            label = label 
            )
        
    osi.set_xlim(0, 1)
    #osi.set_xticks(hranice)
    #osi.set_xticklabels(numpy.array(hranice, dtype=numpy.int_ ))
    osi.set_ylim(0, 1)
    osi.set_xlabel("true positive rate(recall)")
    osi.set_ylabel("precision")
    #osi.axhline(baseline,linestyle = '--')
    osi.legend(loc = "lower right")
    matplotlib.pyplot.tight_layout()
    
    return 0
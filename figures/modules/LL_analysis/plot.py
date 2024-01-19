#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:52:46 2023

@author: jamat
"""

import numpy
import matplotlib
from LL_analysis.calculations import calculate_LSQM

def plot_LL(name0: str, xdata: numpy.dtype(numpy.float_), mse: numpy.dtype(numpy.float_), x_label: str, y_label: str):

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(name0)
    for x,y in zip(xdata,mse):
        #print(x,y)
        zoradene = numpy.argsort(x)
        matplotlib.pyplot.plot(x[zoradene], y[zoradene], label = "")
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    osi.set_ylim(0,1000)
    #matplotlib.pyplot.legend(loc="lower left")
    #name = "{}_LL.png".format(name0)
    #matplotlib.pyplot.savefig(name, dpi= 300)
    #matplotlib.pyplot.close()
    matplotlib.pyplot.show()
    
    return 0

def plot_intervaly(name0: str, xdata: numpy.dtype(numpy.float_), ydata: numpy.dtype(numpy.float_), ydataerror: numpy.dtype(numpy.float_), x_label: str, y_label: str):

    matplotlib.pyplot.figure()
    #matplotlib.pyplot.title(name0)
    posun = numpy.abs(numpy.diff(xdata).mean()/2) # trik, aby som dostal polku intervalu
    intervaly = xdata[:-1] + posun
    matplotlib.pyplot.errorbar(intervaly, ydata, yerr = ydataerror, fmt = "o--" , label = "")
    
    matplotlib.pyplot.axhline(0, linestyle = "--", color = "k")
    
    osi = matplotlib.pyplot.gca()
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    osi.set_xticks(numpy.array(xdata, dtype=numpy.int_))
    osi.set_xticklabels(numpy.array(xdata, dtype=numpy.int_))
    osi.set_ylim(-6,1)
    #matplotlib.pyplot.legend(loc="lower left")
    matplotlib.pyplot.tight_layout()
    name = "{}_slope_vs_intervals.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    name = "{}_slope_vs_intervals.eps".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    matplotlib.pyplot.close()
    matplotlib.pyplot.show()
    
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
    #osi.set_xlim(2.75,3.10)
    #osi.set_ylim(0.55,2.30)
    
    
    osi.text(1.05, 0.00, values,transform=osi.transAxes, fontsize = 'x-large')
    
    matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1), bbox_transform=osi.transAxes, alignment='right' )
    matplotlib.pyplot.tight_layout()
    name = "{}.png".format(name0)
    #matplotlib.pyplot.savefig(name, dpi= 300)
    # matplotlib.pyplot.close()
    matplotlib.pyplot.show()
    
    return 0

def plot_correlaton_all(name0: str, xdata: numpy.dtype(numpy.float_), mse: numpy.dtype(numpy.float_), x_label: str, y_label: str, labels:list[str]):

    lines = ["-", "--", "-.", ":"]
    markers = ["o","s","^","+","."]

    matplotlib.pyplot.figure()
    osi = matplotlib.pyplot.gca()
    
    print(xdata.shape)
    a,b,c = mse.shape
    print(a,b,c)
    
    for i in range(c):
        
        osi.set_prop_cycle(None)
        for x,y in zip(xdata[:,:],mse[:,:,i]):
            print(x,y)
            matplotlib.pyplot.plot(x, y, markers[i])
            
        slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata[:,:]), numpy.ravel(mse[:,:,i]))
        matplotlib.pyplot.axline((0,intercept), slope=slope, color = "k", linestyle = lines[i], label = labels[i])
    
    
    osi.set_xlabel(x_label)
    osi.set_ylabel(y_label)
    
    #xmin,xmax = osi.get_xlim()
    #ymin,ymax = osi.get_ylim()
    
    
    #values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    #osi.set_xlim(xmin,xmax)
    #osi.set_ylim(ymin,ymax)
    osi.set_xlim(2.75,3.10)
    osi.set_ylim(0.55,2.30)
    
    
    #osi.text(1.05, 0.05, values,transform=osi.transAxes)
    
    #matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1), bbox_transform=osi.transAxes, alignment='right' )
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    name = "{}.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300)
    #matplotlib.pyplot.close()
    #matplotlib.pyplot.show()
    
    return 0

def plot_correlaton_all2(name0: str, xdata: numpy.dtype(numpy.float_), rmse_ds: numpy.dtype(numpy.float_), rmse: numpy.dtype(numpy.float_), x_label: str, y_label: str, labels:list[str], labels2:list[str]):

    lines = ["-", "--", "-.", ":"]
    markers = ["o","s","^","+","."]

    #fig, (osi1, osi2) = matplotlib.pyplot.subplots(1, 2, figsize = (12.8,4.8) )# , sharey=True)
    matplotlib.pyplot.figure(figsize = (6.8,2.55), layout = 'constrained')
    #osi1 = matplotlib.pyplot.subplot(223)
    osi1 = matplotlib.pyplot.subplot(121)
    
    for x,y,l in zip(xdata,rmse, labels):
        osi1.plot(x, y, "o", label = l )
    
    osi1.set_xlabel(x_label)
    osi1.set_ylabel(y_label)
    
    xmin,xmax = osi1.get_xlim()
    ymin,ymax = osi1.get_ylim()
    
    slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata), numpy.ravel(rmse))
    osi1.axline((0,intercept), slope=slope, color = "k")
    values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    osi1.set_xlim(2.80,3.10)
    osi1.set_ylim(ymin,ymax)
    
    #osi1.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=matplotlib.pyplot.gcf().transFigure, alignment='right', ncol = 2)
    # rewrite bullets to colors
    handles, text = osi1.get_legend_handles_labels()
    handles2 = []
    for h in handles:
        handles2.append(
            matplotlib.patches.Patch(
                color = h.get_color()
                )
            )
    matplotlib.pyplot.gcf().legend(handles2, text, bbox_to_anchor=(0.05, 1.02, 0.9, .102), loc='lower left',
                      ncols=3, mode="expand", borderaxespad=0.)
    
    osi2 = matplotlib.pyplot.subplot(122)
    
    print(xdata.shape)
    a,b,c = rmse_ds.shape
    print(a,b,c)
    
    for i in range(c):
        
        osi2.set_prop_cycle(None)
        for x,y in zip(xdata[:,:],rmse_ds[:,:,i]):
            print(x,y)
            osi2.plot(x, y, markers[i])
            
        slope, intercept, R2 = calculate_LSQM(numpy.ravel(xdata[:,:]), numpy.ravel(rmse_ds[:,:,i]))
        osi2.axline((0,intercept), slope=slope, color = "k", linestyle = lines[i], label = labels2[i])
    
    
    osi2.set_xlabel(x_label)
    osi2.set_ylabel(y_label)
    
    #xmin,xmax = osi.get_xlim()
    #ymin,ymax = osi.get_ylim()
    
    
    #values = "slope = {slope:.2f} \n$R^2$     = {R2:.3f}".format(slope = slope, R2 = R2)
    
    #osi.set_xlim(xmin,xmax)
    #osi.set_ylim(ymin,ymax)
    osi2.set_xlim(2.80,3.10)
    osi2.set_ylim(0.55,2.30)
    
    
    #osi.text(1.05, 0.05, values,transform=osi.transAxes)
    
    osi2.legend()
    
    name = "{}.png".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300, bbox_inches="tight")
    name = "{}.eps".format(name0)
    matplotlib.pyplot.savefig(name, dpi= 300, bbox_inches="tight")
    #matplotlib.pyplot.close()
    #matplotlib.pyplot.show()
    
    return 0

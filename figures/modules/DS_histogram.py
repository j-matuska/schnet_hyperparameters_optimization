# -*- coding: utf-8 -*-

import matplotlib
import numpy

def plot_histogram(hranice, pocetnost, xlabel, ylabel, labels):
    
    params = {'legend.fontsize': 10,
          'figure.figsize': (3.3, 2.5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
    matplotlib.pyplot.rcParams.update(params)
    
    sirka = (hranice[0][1]-hranice[0][0])/(len(labels)+2)
    
    matplotlib.pyplot.figure()
    
    for j in range(len(labels)):
        matplotlib.pyplot.bar(
            hranice[j][:-1]+((j+1)*sirka),
            pocetnost[j],
            width=sirka, 
            label = '{}'.format(labels[j]),
            log = True
            )
    
    osi = matplotlib.pyplot.gca()
    osi.set_xlim(hranice[0][0],hranice[0][-1])
    osi.set_xticks(hranice[0])
    osi.set_xticklabels(numpy.array(hranice[0], dtype=numpy.int_))
    osi.set_xlabel(xlabel)
    #osi.set_ylim(1e-6,1)
    osi.set_ylabel(ylabel)
    pismo = matplotlib.font_manager.FontProperties(fname="/home/jamat/Downloads/cmu.serif-italic.ttf")
    tmp = matplotlib.pyplot.legend(bbox_to_anchor=(-0.1, 1.02, 1.0, .102), loc='lower left',
                      ncols=2, mode="expand", borderaxespad=0., prop=pismo)
        
    matplotlib.pyplot.tight_layout()
    
    name = "{}_distribution_density.png".format("DS")
    matplotlib.pyplot.savefig(name, dpi= 300)
    name = "{}_distribution_density.eps".format("DS")
    matplotlib.pyplot.savefig(name, dpi= 300)
    
    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)
    
    return 0

def plot_histogram_broken_axis(hranice, pocetnost, xlabel, ylabel, labels):
    
    params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
    matplotlib.pyplot.rcParams.update(params)
    
    sirka = (hranice[0][1]-hranice[0][0])/(len(labels)+2)
    
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)  # adjust space between axes
    
    for j in range(len(labels)):

        # plot the same data on both axes
        ax1.bar(
            hranice[j][:-1]+((j+1)*sirka),
            pocetnost[j],
            width=sirka, 
            label = '{}'.format(labels[j].replace('_', ' ') )
            )
        ax2.bar(
            hranice[j][:-1]+((j+1)*sirka),
            pocetnost[j],
            width=sirka, 
            label = '{}'.format(labels[j])
            )

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(0.01, 0.3)  # outliers only
    ax2.set_ylim(0, 0.0026)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    ax2.set_xlim(hranice[0][0],hranice[0][-1])
    ax2.set_xticks(hranice[0])
    #ax1.set_xlim(hranice[0][0],hranice[0][-1])
    #ax1.set_xticks(hranice[0])
    ax2.set_xticklabels(hranice[0])
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax1.legend()#prop={'weight':'bold'})
    
    matplotlib.pyplot.tight_layout()
    
    name = "{}_distribution_density_BA.png".format("DS")
    matplotlib.pyplot.savefig(name, dpi= 300)

    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)
    
    return 0
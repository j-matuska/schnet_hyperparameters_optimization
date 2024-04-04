#!/usr/bin/env python3

"""
    Calculation of MSE dependence on DS from csv for five different splits 
    designated as 01 02 03 04 05. 
    Each argument can contain string XX. This string is then replaced 
    by 01 02 03 04 05 in sequence
    E.g.:
        python3 analyze_csv_multiple_splits.py ./80_XX/ in_vivo_test_80_XX.csv

    
    Arguments:
    ----------
    first: string
        path to folder where are stored results. 
    second: string
        name of the csv file

    Returns
    -------
    file : PNG
        Containin 2 figures: MSE vs DS; sigma,mi vs DS.
    file: TXT
        Contain source data to plot investigated data

"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot
import logging

error = 0.0
mse = 0.0

def main():
    
    params = {'legend.fontsize': 10,
          #'figure.figsize': (15, 5),
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
    matplotlib.pyplot.rcParams.update(params)
    
    logging.basicConfig(level = logging.INFO, filename='analyze.log', filemode='a', force=True)
    logging.info(' ============================================= \n' )

    DS_type = "DS" 
    DS_type_label = '$\mathrm{DS_{expected}}$ [kcal/mol]'
    #TF
    prefix_path = sys.argv[1]
    suffix_path  = sys.argv[2]
    
    run(prefix_path, suffix_path, DS_type, DS_type_label)
    
    matplotlib.pyplot.rcParams.update(matplotlib.pyplot.rcParamsDefault)

    return 0


def run(prefix_path, suffix_path, DS_type, DS_type_label):
    
    (mse, sigma,
     intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_over_2_ds
     ) = load_results(prefix_path, suffix_path )
    # ploting
    NN, dataset = extract_keywords(prefix_path, suffix_path)
    save_plots(mse, sigma,
               intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds,
               "", suffix_path, DS_type_label)
    # saving
    DS_savetxt(intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_over_2_ds, mse, sigma, suffix_path+DS_type)
    # save verification
    (stred_intervalu, rel_pocet_ds_avg, rel_pocet_ds_std, mse_ds_avg, mse_ds_std, mu_ds_avg, mu_ds_std,
     sigma_ds_avg, sigma_ds_std, fraction_over_2_ds_avg, fraction_over_2_ds_std) = loadtxt(suffix_path+DS_type)
    print(stred_intervalu, rel_pocet_ds_avg, rel_pocet_ds_std, mse_ds_avg, mse_ds_std, mu_ds_avg, mu_ds_std, sigma_ds_avg, sigma_ds_std)
    
    return 0


def extract_keywords(prefix_path, suffix_path):
    
    NN = prefix_path[:2]
    dataset = os.path.splitext((os.path.split(suffix_path)[1]))[0]
    
    return NN, dataset


def save_plots(mse, sigma, 
               intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds,
               prefix_path, suffix_path, DS_type_label):
    
    # DS vs properties
    DS_plot(intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, mse, sigma, DS_type_label)    
    png_name=str(prefix_path+suffix_path+"_partial_Gaussians_stat_DS.png")
    matplotlib.pyplot.savefig(png_name, dpi= 300)
    png_name=str(prefix_path+suffix_path+"_partial_Gaussians_stat_DS.eps")
    matplotlib.pyplot.savefig(png_name, dpi= 300)  

    return 0


def load_results(prefix_path_to_result, suffix_path_to_result):
    
    
    adresare = ['01', '02', '03', '04', '05']
    sablona =prefix_path_to_result+suffix_path_to_result

    meno_all = []
    AD_all = []
    TF_all = []
    
    mse_all = []
    sigma_all = []
    
    meno_xyz = []
    
    intervaly_all = []
    mse_ds_all = []
    rel_pocet_ds_all = []
    mu_ds_all = []
    sigma_ds_all = []
    fraction_over_2_ds_all = []
    
    # data agregation
    
    for adresar in adresare:
        
        meno, AD, TF = read_csv(sablona.replace("XX",adresar))
        meno_all.append(meno)
        AD_all.append(AD)
        TF_all.append(TF)
        
        error, mse, sigma = errors(meno, AD, TF)
        mse_all.append(mse)
        sigma_all.append(sigma)
        
        meno_xyz.append(meno)
        
        AD_min=np.floor(np.amin(AD))
        AD_max=np.ceil(np.amax(AD))
        AD_min=-15
        AD_max=1
        print(AD_min, AD_max)
        
        intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_over_2_ds = DS_gaussian(np.linspace(AD_min, AD_max, 9), AD, error) # just to compare to PR
        intervaly_all.append(intervaly)
        mse_ds_all.append(mse_ds)
        rel_pocet_ds_all.append(rel_pocet_ds)
        mu_ds_all.append(mu_ds)
        sigma_ds_all.append(sigma_ds)
        fraction_over_2_ds_all.append(fraction_over_2_ds)
        
    
    mse = np.asfarray(mse_all)
    sigma = np.asfarray(sigma_all)
    
    intervaly = np.array(intervaly_all)
    mse_ds = np.asfarray(mse_ds_all)
    rel_pocet_ds = np.asfarray(rel_pocet_ds_all)
    mu_ds = np.asfarray(mu_ds_all)
    sigma_ds = np.asfarray(sigma_ds_all)
    fraction_over_2_ds = np.asfarray(fraction_over_2_ds_all)
    
    print("MSE(01-05): {} (kcal/mol)^2".format(mse))
    print("sigma(01-05): {} kcal/mol".format(sigma))
    
    logging.info("MSE(01-05): {} (kcal/mol)^2".format(mse))
    logging.info("sigma(01-05): {} kcal/mol".format(sigma))
    
    logging.info("MSE: {} +/- {} (kcal/mol)^2".format(mse.mean(),mse.std(ddof=0)))

    return (mse, sigma, 
            intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_over_2_ds
            )


def read_csv(name_csv):
    
    df = pd.read_csv(name_csv, sep=';')
    data = df.values.transpose()

    c = data[0] # coumpounds labels
    x = data[1]
    y = data[2]
    
    print("Number of molecules in {}: {}".format(name_csv, c.size))
    
    original = x.astype(float)
    predicted = y.astype(float)
    
    return c,original,predicted


def errors(name, original, predicted):
    
    error = np.subtract(original,predicted)
    se = np.square(error)
    mse = round(se.mean(),3)
    sigma = np.std(error,ddof=1)
    
    return error, mse, sigma

 
def DS_gaussian(intervaly, original, error):   
    
    # MAE
    ae = np.absolute(error)
    

    sigma_ds = np.zeros(intervaly.size-1,dtype=np.float_)
    mu_ds = np.zeros(intervaly.size-1,dtype=np.float_)
    mse_ds = np.zeros(intervaly.size-1,dtype=np.float_)
    pocet_ds = np.zeros(intervaly.size-1,dtype=np.int_)
    fraction_pocet_ds = np.zeros(intervaly.size-1,dtype=np.float_)
    rel_pocet_ds = np.zeros(intervaly.size-1,dtype=np.float_)
    
    for i in np.arange(intervaly.size-1):
        
        umiestnenie_bool = np.logical_and(original > intervaly[i], original <= intervaly[i+1])
        if i == 0: # extension of the first interval do -inf
            doplnok = (original < intervaly[i])
            umiestnenie_bool[doplnok] = True
        if i == intervaly.size-2: # extension of the last interval do inf
            doplnok = (original > intervaly[i+1])
            umiestnenie_bool[doplnok] = True
        umiestnenie = np.where(umiestnenie_bool)
        rel_pocet_ds[i] = umiestnenie[0].size/umiestnenie_bool.size    
        sigma_ds[i] = np.std(error[umiestnenie],ddof=0)
        mu_ds[i] = np.mean(error[umiestnenie])
        mse_ds[i] = np.square(error[umiestnenie]).mean()
        pocet_ds[i] = np.count_nonzero(ae[umiestnenie] > 2.0)
        fraction_pocet_ds[i] = pocet_ds[i]/umiestnenie[0].size

 
    return intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_pocet_ds

def DS_plot(intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, mse, sigma, variable_label):
    
    intervaly_avg = intervaly.mean(axis=0)
    for i in range(5):
        if not np.array_equal(intervaly_avg, intervaly[i,:]):
            print("Warning: arrays do not fit together")
            print(intervaly_avg[:5],intervaly[i,:5])
    
    mse_ds_avg = mse_ds.mean(axis=0, where=np.isfinite(mse_ds))
    rel_pocet_ds_avg = rel_pocet_ds.mean(axis=0)
    mu_ds_avg = mu_ds.mean(axis=0, where=np.isfinite(mu_ds))
    sigma_ds_avg = sigma_ds.mean(axis=0, where=np.isfinite(sigma_ds))
    
    mse_ds_std = mse_ds.std(axis=0,ddof=1, where=np.isfinite(mse_ds))
    rel_pocet_ds_std = rel_pocet_ds.std(axis=0,ddof=1)
    mu_ds_std = mu_ds.std(axis=0,ddof=1, where=np.isfinite(mu_ds))
    sigma_ds_std = sigma_ds.std(axis=0,ddof=1, where=np.isfinite(sigma_ds))
    
    mse_avg = mse.mean(axis=0)
    sigma_avg = sigma.mean(axis=0)
    
    posun = np.abs(np.diff(intervaly_avg).mean()/2) 
    
    print(matplotlib.get_backend())
    
    fig, ( ax_error3, ax_error4) = matplotlib.pyplot.subplots(2, 1, figsize=(3.3,4.4), sharex=True)
    fig.subplots_adjust(left= 0.17, bottom=0.11, right=0.84, top=0.95, hspace=0)
    print(matplotlib.rcParams["figure.subplot.top"])

    print("Fraction check: {}".format(rel_pocet_ds.sum()))
    
    color = 'tab:blue'
    ax_error3.set_ylabel(r'MSE [$\mathrm{(kcal/mol)^2}$]', color = color)
    ax_error3.errorbar(intervaly_avg[:-1]+posun, mse_ds_avg, yerr = mse_ds_std, fmt = 'o--', color = color)
    ax_error3.set_xlim(intervaly_avg[0],intervaly_avg[-1])
    ax_error3.set_xticks(intervaly_avg)
    ax_error3.set_xticklabels(np.array(intervaly_avg, dtype=np.int_))
    ax_error3.set_ylim(0,5)
    ax_error3.axhline(mse_avg,linestyle='--',color = color)
    
    ax_error3b = ax_error3.twinx()
    color = 'tab:green'
    ax_error3b.set_ylabel("rel. abundance", color = color)
    ax_error3b.errorbar(intervaly_avg[:-1]+posun, rel_pocet_ds_avg, yerr = rel_pocet_ds_std, fmt = 's:', color = color, label = "distribution")
    ax_error3b.set_ylim(0,1.0)
    
    ax_error4.set_xlabel(r'{}'.format(variable_label))
    color = 'tab:blue'
    ax_error4.set_ylabel("$\mu$ [kcal/mol]", color = color)
    ax_error4.errorbar(intervaly_avg[:-1]+posun, mu_ds_avg, yerr = mu_ds_std, fmt = 's--', color = color)
    ax_error4.axhline(0.0, linestyle='--', color = color)
    updown_lim = 2.5 
    ax_error4.set_ylim(-updown_lim, updown_lim)
    delenie = ax_error4.get_yticks()
    ax_error4.set_yticks(delenie[1:-1])
    ax_error4b=ax_error4.twinx()
    color = 'tab:green'
    ax_error4b.set_ylabel("$\sigma$ [kcal/mol]", color = color)
    ax_error4b.errorbar(intervaly_avg[:-1]+posun, sigma_ds_avg, yerr = sigma_ds_std, fmt = 'o-.', color = color)
    ax_error4b.axhline(sigma_avg, linestyle='-.', color = color)
    ax_error4b.set_ylim(0,5)
    delenie = ax_error4b.get_yticks()
    ax_error4b.set_yticks(delenie[0:-1])
    
    return 0

def DS_savetxt(intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, fraction_over_2_ds, mse, sigma, variable_label):
    
    intervaly_avg = intervaly.mean(axis=0)
    for i in range(5):
        if not np.array_equal(intervaly_avg, intervaly[i,:]):
            print("Warning: arrays do not fit together")
            print(intervaly_avg[:5],intervaly[i,:5])
    
    mse_ds_avg = mse_ds.mean(axis=0, where=np.isfinite(mse_ds))
    rel_pocet_ds_avg = rel_pocet_ds.mean(axis=0)
    mu_ds_avg = mu_ds.mean(axis=0, where=np.isfinite(mu_ds))
    sigma_ds_avg = sigma_ds.mean(axis=0, where=np.isfinite(sigma_ds))
    fraction_over_2_ds_avg = fraction_over_2_ds.mean(axis=0, where=np.isfinite(fraction_over_2_ds))
    
    mse_ds_std = mse_ds.std(axis=0,ddof=1, where=np.isfinite(mse_ds))
    rel_pocet_ds_std = rel_pocet_ds.std(axis=0,ddof=1)
    mu_ds_std = mu_ds.std(axis=0,ddof=1, where=np.isfinite(mu_ds))
    sigma_ds_std = sigma_ds.std(axis=0,ddof=1, where=np.isfinite(sigma_ds))
    fraction_over_2_ds_std = fraction_over_2_ds.std(axis=0, ddof=1, where=np.isfinite(fraction_over_2_ds))
    
    posun = np.abs(np.diff(intervaly_avg).mean()/2) 
    data = np.vstack((intervaly_avg[:-1]+posun,
                         rel_pocet_ds_avg, rel_pocet_ds_std,
                         mse_ds_avg, mse_ds_std,
                         mu_ds_avg, mu_ds_std,
                         sigma_ds_avg, sigma_ds_std,
                         fraction_over_2_ds_avg, fraction_over_2_ds_std)).transpose()
    
    np.savetxt("calibration_{}.txt".format(variable_label), data, header = "{:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}, {:<25}".format("stred_intervalu", "relativny_pocet", "relativny_pocet_std", "mse", "mse_std", "mu", "mu_std", "sigma", "sigma_std", "fraction over 2kcal/mol", "fraction over 2kcal/mol std"))
    
    return 0


def loadtxt(variable_label):
    
    data = np.loadtxt("calibration_{}.txt".format(variable_label))
    stred_intervalu, rel_pocet_ds_avg, rel_pocet_ds_std, mse_ds_avg, mse_ds_std, mu_ds_avg, mu_ds_std, sigma_ds_avg, sigma_ds_std, fraction_over_2_ds_avg, fraction_over_2_ds_std = np.vsplit(data.transpose(),11)
    
    return stred_intervalu, rel_pocet_ds_avg, rel_pocet_ds_std, mse_ds_avg, mse_ds_std, mu_ds_avg, mu_ds_std, sigma_ds_avg, sigma_ds_std, fraction_over_2_ds_avg, fraction_over_2_ds_std


if __name__ == "__main__":
   main()

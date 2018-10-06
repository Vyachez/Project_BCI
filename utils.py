#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:27:02 2018

Functions to process data for OpenBCI brainwaives reading

@author: vyachez
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys; sys.path.append('..') 

# dropping unnecessary staff
def dip_drop(dset):
    dset = dset.drop(columns=[0,9,10,11])
    dset = dset.reset_index(drop=True)
    print(dset.shape)
    return dset

# preparing basics    
def basics(dset):
    '''cleaning up time column'''
    # converting timestamp into seconds (cutting off milisecs)
    dset[12] = dset[12].str[1:-4]
    dset = dset.rename(columns={12: "sec"})
    print('Realigned time column')
    return dset

# Cleaning data from spikes to prepare for normalisation
# iterating through each second to clean-up spikes of data
def variance_clean(dset, var):
    """ dset - dataset to cleanup
        var - max variance - all that above is to catch and remove
    """
    for chan in range(8):
        for sec in np.unique(dset['sec']):
            min_edge = min(dset.loc[dset['sec'] == sec][chan+1])
            max_edge = max(dset.loc[dset['sec'] == sec][chan+1])
            variance = max_edge - min_edge
            idx = dset.loc[dset[chan+1] == max_edge].index[0]
            if variance > var:
                #print('Channel {} | Second {} | Index {} | Variance:] = {}'.format(chan+1, sec, idx, variance))
                dset = dset.drop(index=dset.loc[dset['sec'] == sec].index)
                # reseting the index
                dset = dset.reset_index(drop=True)
                #print('Dropped')
    print('Cleaned spikes larger than', var)
    return dset   

# balancing intervals to set length (optimal number of rows)
def balance_intervals(dset, int_no):
    """ dset - dataset to cleanup
        int_no - min length of intervals within one second"""
    idarr = np.array([[i, len(dset['sec'].loc[dset['sec'] == i])] for i in np.unique(dset['sec'])])
    for i in idarr:
        if int(i[1]) < int_no:
            date = i[0]
            # removing short/incomplete
            dset = dset.drop(index=dset.loc[dset['sec'] == date].index)
        elif int(i[1]) > int_no:
            date = i[0]
            end_ind = dset.loc[dset['sec'] == date].index[-1]
            cut_ind = dset.loc[dset['sec'] == date].index[int_no-1]
            # cutting excessive
            dset = dset.drop(dset.index[cut_ind:end_ind])
        dset = dset.reset_index(drop=True)
    return dset

# see if all of intervals have the same length
def balance_check(dset, dur):
    for i in np.unique(dset['sec']):
        if len(dset['sec'].loc[dset['sec'] == i]) != dur:
            print('Seconds are not equal!')
            print(np.array([i, len(dset['sec'].loc[dset['sec'] == i])]))
    print("Check completed for balanced intervals!")
    return True

# checking number of seconds in dataset(unique)
def seconds(dset):
    '''takes dataset as a  argument and utputs
    number of unique seconds'''
    print("\nSeconds in dataset now: ", len(np.unique(dset['sec'])))
    
# see if all of intervals have the same length by second
def sec_disp(dset):
    return np.array([[i, len(dset['sec'].loc[dset['sec'] == i])] for i in np.unique(dset['sec'])])

# scaling function
def scaler(dset, secs, dur):
    '''Scaling function takes dataset of 8 channels
    and returns rescaled dataset.
    Rescales by interval (second).
    arg 'secs' is number of seconds to take into one scaling tensor.
    Scaling is between 1 and 0
    All datasets for training should be equalized'''
    # first - getting length of dataset modulo number of seconds for scaling
    intlv = secs*dur
    if len(dset['sec'])/intlv > 1:
        lendset = int(len(dset['sec'])/intlv)
        dset = dset[0:lendset*intlv]
        dset = dset.reset_index(drop=True)
    else:
        print("Inappropriate length of dataset. Too short. Set up less seconds for batch or choose another dataset.")
    seconds(dset)
    # now scaling
    if balance_check(dset, dur=dur):
        for chan in range(8):
            for i in range(int(len(dset['sec'])/intlv)):
                tmpdat = dset.loc[i*intlv:i*intlv+intlv-1, chan+1]
                tmpdat = (tmpdat-min(tmpdat))/(max(tmpdat)-min(tmpdat))  
                dset.loc[i*intlv:i*intlv+intlv-1, chan+1]= tmpdat
        dset = dset.reset_index(drop=True)
        print("Dataset has been rescaled \n")
        return dset
    else:
        print("\nDataset intervals are not balanced! Check the code and functions order.")

# equalizing datasets by seconds' intervals
def equalizing(r_dat, w_dat):
    '''Taking two datasets and equalising its lengths
    To fit training algorithms'''
    # taking length of datasets
    r_dat_sec = len(np.unique(r_dat['sec']))
    w_dat_sec = len(np.unique(w_dat['sec']))
    # eualizing
    if r_dat_sec > w_dat_sec:
        min_len = min(r_dat_sec, w_dat_sec)
        for i in range(min_len+1,r_dat_sec+1):
            r_dat = r_dat.drop(index=r_dat.loc[r_dat['sec'] == np.unique(r_dat['sec'])[-0]].index)
            #print('Dropped second from r_dat')
    elif w_dat_sec > r_dat_sec:
        min_len = min(r_dat_sec, w_dat_sec)
        for i in range(min_len+1,w_dat_sec+1):
            w_dat = w_dat.drop(index=w_dat.loc[w_dat['sec'] == np.unique(w_dat['sec'])[-0]].index)
            #print('Dropped second from r_dat')
    else:
        print('Seconds are equal!') 
    r_dat = r_dat.reset_index(drop=True)
    w_dat = w_dat.reset_index(drop=True)
    print("Equalized!")
    # checking number of unique seconds
    r_dat_sec = len(np.unique(r_dat['sec']))
    w_dat_sec = len(np.unique(w_dat['sec']))
    print("\nSeconds: r_dat:", r_dat_sec)
    print("Seconds: w_dat:", w_dat_sec)
    print("\nr_dat dim:", r_dat.shape)
    print("w_dat dim:", w_dat.shape)
    return r_dat, w_dat

# plotting function
def d_plot(dset, chan=0, seconds=0, start=0, dur=0):
    '''this can plot each single chanel and within
    certain number of seconds'''
    sec = np.array(dset.index)
    plt.figure(figsize=(20,5))
    if chan != 0:
        if seconds != 0:
            intv = dur*start
            intv_1 = intv+dur*seconds
            sec = np.array(dset.index[intv:intv_1])
            plt.plot(sec, np.array(dset[chan][intv:intv_1]), label='ch')
            plt.legend()
            _ = plt.ylim()
        else:
            plt.plot(sec, np.array(dset[chan]), label='ch')
            plt.legend()
            _ = plt.ylim()
    else:
        if seconds != 0:
            intv = dur*start
            intv_1 = intv+dur*seconds
            sec = np.array(w_dat.index[intv:intv_1])
            plt.plot(sec, np.array(dset[1][intv:intv_1]), label='ch1')
            plt.plot(sec, np.array(dset[2][intv:intv_1]), label='ch2')
            plt.plot(sec, np.array(dset[3][intv:intv_1]), label='ch3')
            plt.plot(sec, np.array(dset[4][intv:intv_1]), label='ch4')
            plt.plot(sec, np.array(dset[5][intv:intv_1]), label='ch5')
            plt.plot(sec, np.array(dset[6][intv:intv_1]), label='ch6')
            plt.plot(sec, np.array(dset[7][intv:intv_1]), label='ch7')
            plt.plot(sec, np.array(dset[8][intv:intv_1]), label='ch8')
            plt.legend()
            _ = plt.ylim()
        else:
            plt.plot(sec, np.array(dset[1]), label='ch1')
            plt.plot(sec, np.array(dset[2]), label='ch2')
            plt.plot(sec, np.array(dset[3]), label='ch3')
            plt.plot(sec, np.array(dset[4]), label='ch4')
            plt.plot(sec, np.array(dset[5]), label='ch5')
            plt.plot(sec, np.array(dset[6]), label='ch6')
            plt.plot(sec, np.array(dset[7]), label='ch7')
            plt.plot(sec, np.array(dset[8]), label='ch8')
            plt.legend()
            _ = plt.ylim()
            
# data vizualizer preparation
def vizualize_prep(fl):
    '''prepares dataset for vizuzlisation without any cleaning
    "fl" (fullpath) - Takes file and returns balanced preprocessed data'''
    # loading the data
    dset = pd.read_csv(fl, sep=",", header=None)
    # removing unnecessary columns
    dset = dip_drop(dset)
    # realigning timelapse column
    dset = basics(dset)
    return dset
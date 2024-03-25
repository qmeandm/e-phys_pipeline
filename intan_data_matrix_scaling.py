#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:03:13 2024

@author: xander
"""



import os
import matplotlib as plt
import numpy as np
from scipy.signal import bessel, lfilter, iirnotch, sosfreqz, sos2tf, convolve

# ----------------------------------------------------------------------------


os.chdir('/Users/xander/load-rhs-notebook-python/')

from importrhsutilities import *

exec(open("importrhsutilities.py").read())

# file notes: 
    # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
    #  - 20uA

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155222.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)



# ----------------------------------------------------------------------------------------
# below this line begin messing around with manipulating data directly
# ----------------------------------------------------------------------------------------

# NOTE: Channel name input is manual for now. You must edit the dictionary position in the
#       rawvolts variable such that Channel D-005 ---> [1]... D-010 --> [6]

# Find the text notes in the INTAN file that give the stim parameters, make a str:
stim_set = str(result['notes']['note1'])
stim_amps = stim_set[:5]

# What channel do you want to analyze data from? INPUT MANUALLY!
nanoclip_channel = 5 # n-1 of the real clip channel! e.g. 0 = acutal channel 1

channel_name = str(nanoclip_channel +1) # set the channel name to call it in the raw data below


# Pull data from the result dictionary in importrhsutilities.py:
    
raw_volts = np.array(result['amplifier_data'][nanoclip_channel]) # Channel D-005 -> [0]... D-010 -> [5]
time = np.array(result['t'])

# Plot the raw data:

# plt.figure(figsize=(10, 6))
# plt.plot(time, raw_volts)
# plt.show
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (undefined units)")
# # plt.xlim(2.800, 2.900)
# # plt.ylim(-0.500, 0.500)
# plt.title("Raw Amplifier Data, Stim = " + stim_amps + ', Channel = ' + channel_name)

# ----------------------------------------------------------------------------------------
# below this line scale the raw data matrix as in the MATLAB script and adjust the offset
# ----------------------------------------------------------------------------------------

scaled_data = [0.195 * (z - 32768) for z in raw_volts] # claim: makes units microvolts

common_offset = sum(scaled_data) / len(scaled_data)

print("Common Offset:", common_offset)

no_offset_data = [a - common_offset for a in scaled_data]

# plt.figure(figsize=(10, 6))
# plt.plot(time, no_offset_data, color='purple')
# plt.show
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (uV)")
# plt.xlim(0.0, 5.0)
# # plt.ylim(-30, 30)
# plt.title("Scaled Amplifier Data, Channel = " + channel_name)

# ----------------------------------------------------------------------------------------
# below this line try thresholding data after offsetting
# ----------------------------------------------------------------------------------------

# # Apply a threshold to the data as a numpy array:
# threshold = 100  # microvolts (???)

# data_arr = np.array(no_offset_data)

# # thresh_signal_arr = np.where(abs(data_arr) > threshold, 0, data_arr)

# # thresh_signal_list = thresh_signal_arr.tolist()

# data_arr[data_arr > threshold] = 0
# data_arr[data_arr < -threshold] = 0

# thresh_data = data_arr

# # Plot the original and thresholded signals
# plt.figure(figsize=(10, 6))
# plt.plot(time, raw_volts, label='Original Signal')
# plt.plot(time, thresh_data, label='Thresholded Signal')
# plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
# plt.xlim(1.0, 10.0)
# plt.ylim(-150, 150)
# plt.title('Thresholding = ' + str(threshold)+' uV, '+ 'Channel = '+channel_name)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage')
# plt.legend()
# plt.show()


# ----------------------------------------------------------------------------------------
# below this line try using a bessel bandpass filter as in Otchy's nanoclip paper
# ----------------------------------------------------------------------------------------

# Define filter parameters:
fs = 30000  # Sampling frequency
f1 = 1  # Lower cutoff frequency
f2 = 10000  # Upper cutoff frequency
order = 4  # Filter order, 2 to 4 has been working best

# Design the Bessel bandpass filter:
b, a = bessel(order, [f1 / (fs / 2), f2 / (fs / 2)], 'bandpass')

# Delpoy the filter on the thresholded data AND raw data:

bandpass_signal = lfilter(b, a, thresh_data)
bandpass_raw = lfilter(b, a, raw_volts)

# Plot the comparison of the filtered data and the raw data:

plt.figure(figsize=(10, 6))
plt.plot(time, raw_volts, label='Original Signal')
plt.plot(time, bandpass_signal, label='Original Signal + Bessel Bandpass Filter')
# plt.plot(time, bandpass_signal, color='red', label='Threshold + Bessel Bandpass Filter')
plt.title('Threshold + Bessel Bandpass Filter, 100uA Stim, Channel = ' + channel_name)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV???)')
plt.xlim(1.808, 1.818)  # Set the x-axis limits
plt.ylim(-150, 150)  # Set the y-axis limits
plt.legend()
plt.show()



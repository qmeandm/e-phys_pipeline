#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:38:10 2024

@author: xander

Script to use Intan's python code to read in .rhs files and parse them for data,
then plot whatever e-phys data we would like.

TO DO:
    - file reader loop
    - spike counter
    - zero crossing finder 
        - for multiple spikes in a given recording
        - then averages those and stores that spike as a single vector
        - make a counter to find the number of spikes in a given file - FIND A CUTOFF (10, 15?)
    - spike alignment / batch plotting

OPEN QUESTIONS: 
    - are we going to average X number of spikes for a given stim current recording?
        - Otchy uses n = 16 (Fig 3d)
    
    - 



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

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155256.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)

channel_name = 'D-009' # Change this variable and re-run cell to plot a different channel

# if data_present:
#     plot_channel(channel_name, result)
    
# else:
#     print('Plotting not possible; no data in this file')
    
# ----------------------------------------------------------------------------------------
# below this line begin messing around with manipulating data directly
# ----------------------------------------------------------------------------------------

# NOTE: Channel name input is manual for now. You must edit the dictionary position in the
#       rawvolts variable such that Channel D-005 ---> [1]... D-010 --> [6]

# Find the text notes in the INTAN file that give the stim parameters, make a str
stim_set = str(result['notes']['note1'])
stim_amps = stim_set[:5]

rawvolts = np.array(result['amplifier_data'][1]) # Channel D-005 ---> [1]... D-010 --> [6]
time = np.array(result['t'])

y = rawvolts.tolist()
x = time.tolist()

plt.plot(x, y)

plt.show
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
# plt.xlim(2.885, 2.895)
# plt.ylim(-0.500, 0.500)
plt.title("Raw Amplifier Data")


# ----------------------------------------------------------------------------------------
# below this line try thresholding data before filtering
# ----------------------------------------------------------------------------------------

# Apply a threshold to the data as a numpy array:
threshold = 0.4  # 4 millivolts (???)

orig_data_arr = np.array(y)

thresh_signal_arr = np.where(abs(orig_data_arr) > threshold, 0, orig_data_arr)

thresholded_signal_list = thresh_signal_arr.tolist()

# OLD thresholding attempt, maybe doesn't work, keep commented: 
# thresholded_signal_list = [np.sign(sample) * min(np.abs(sample), threshold) for sample in y]

# Convert the thresholded list back to a NumPy array:
# thresholded_signal = np.array(thresholded_signal_list)

# Plot the original and thresholded signals
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Signal')
plt.plot(x, thresholded_signal_list, label='Thresholded Signal')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.xlim(2.0, 5.0)
plt.ylim(-0.500, 0.500)
plt.title('Thresholding - ' + str(threshold)+' V')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------
# below this line try using a bessel bandpass filter as in Otchy's nanoclip paper
# ----------------------------------------------------------------------------------------

# Define filter parameters:
fs = 30000  # Sampling frequency
f1 = 1  # Lower cutoff frequency
f2 = 10000  # Upper cutoff frequency
order = 4  # Filter order

# Design the Bessel bandpass filter:
b, a = bessel(order, [f1 / (fs / 2), f2 / (fs / 2)], 'bandpass')

# Delpoy the filter on the thresholded data:

bandpass_signal = lfilter(b, a, thresholded_signal_list)

plt.figure(figsize=(10, 6))
# plt.plot(time, signal, label='Original Signal')
plt.plot(time, bandpass_signal, color='red', label='Threshold + Bessel Bandpass Filter')
plt.title('Threshold + Bessel Bandpass Filter, 110uA Stim, ' + channel_name)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.xlim(2.840, 2.880)  # Set the x-axis limits
plt.ylim(-0.500, 0.500)  # Set the y-axis limits
plt.legend()
plt.show()


# Deploy the Bessel bandpass filter on the RAW data:

# raw_bandpass_signal  = lfilter(b, a, y)

# plt.figure(figsize=(10, 6))
# # plt.plot(time, signal, label='Original Signal')
# plt.plot(time, raw_bandpass_signal, color='blue', label='Raw Data + Bessel Bandpass Filter')
# plt.title('Raw Data + Bessel Bandpass Filter, 110uA Stim, ' + channel_name)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.xlim(2.0, 4.880)  # Set the x-axis limits
# plt.ylim(-5, 5)  # Set the y-axis limits
# plt.legend()
# plt.show()



# Try to run the low pass + high pass bessel filters on the thresholded signal. 
# THIS CURRENTLY FAILS, DO NOT USE:

fs = 30000 # default sampling frequency = 30 kHz


# Design a Bessel low-pass filter:
low_pass_cutoff = 300  # Cutoff frequency in Hz FROM INTAN SETTINGS
low_pass_order = 4    # Filter order FROM INTAN SETTINGS
low_pass_b, low_pass_a = bessel(N=low_pass_order, Wn=low_pass_cutoff / (0.5 * fs), btype='low')

# Design a Bessel high-pass filter:
high_pass_cutoff = 3000   # Cutoff frequency in Hz FROM INTAN SETTINGS
high_pass_order = 4   # Filter order FROM INTAN SETTINGS
high_pass_b, high_pass_a = bessel(N=high_pass_order, Wn=high_pass_cutoff / (0.5 * fs), btype='high')


# Combine the filters via convolution:
combined_b = convolve(low_pass_b, high_pass_b)
combined_a = convolve(low_pass_a, high_pass_a)

DUAL_filtered_signal = lfilter(combined_b, combined_a, thresholded_signal_list)


plt.figure(figsize=(10, 6))
# plt.plot(time, signal, label='Original Signal')
plt.plot(time, DUAL_filtered_signal, color='red', label='Combined Bessel Filters')
plt.title('Bessel Low-Pass and High-Pass Filters, 110uA Stim, ' + channel_name)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.xlim(1.0, 7.00)  # Set the x-axis limits
plt.ylim(-0.500, 0.500)  # Set the y-axis limits
plt.legend()
plt.show()



# ----------------------------------------------------------------------------------------
# below this line begin implementing filters individually
# NOTE: the 60Hz Notch filter MAY HAVE been automatically applied by the RHS file...
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>CHECK THIS!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ----------------------------------------------------------------------------------------


# -------------------- first, LOW PASS filter ---------------------------------

# fs = 30000 # default sampling frequency = 30 kHz

# # Design a Bessel low-pass filter
# low_pass_cutoff = 300  # Cutoff frequency in Hz FROM INTAN SETTINGS
# low_pass_order = 4    # Filter order FROM INTAN SETTINGS
# low_pass_b, low_pass_a = bessel(N=low_pass_order, Wn=low_pass_cutoff / (0.5 * fs), btype='low')


# signal = y # defined above, this is the RAW voltage signal converted to a list

# # Apply the filter to the signal
# LPF_filtered_signal = lfilter(low_pass_b, low_pass_a, signal)

# # Plot the original and filtered signals
# plt.figure(figsize=(10, 6))
# plt.plot(time, signal, label='Original Signal')
# plt.plot(time, LPF_filtered_signal, label='Filtered Signal')
# plt.title('Low-Pass Filter ONLY, Bessel N=4')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (mV)')
# plt.xlim(2, 4)  # Set the x-axis limits
# plt.ylim(-15, 15)  # Set the y-axis limits
# plt.legend()
# plt.show()

# # ----------------------- now HIGH PASS filter --------------------------------


# # Design a Bessel high-pass filter:
    
# high_pass_cutoff = 3000   # Cutoff frequency in Hz FROM INTAN SETTINGS
# high_pass_order = 4   # Filter order FROM INTAN SETTINGS
# high_pass_b, high_pass_a = bessel(N=high_pass_order, Wn=high_pass_cutoff / (0.5 * fs), btype='high')

# HPF_filtered_signal = lfilter(high_pass_b, high_pass_a, signal)


# # Plot the original and filtered signals:
    
# plt.figure(figsize=(10, 6))
# plt.plot(time, signal, label='Original Signal')
# plt.plot(time, HPF_filtered_signal, label='Filtered Signal')
# plt.title('High-Pass Filter ONLY, Bessel N=4')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (mV)')
# plt.xlim(2, 4)  # Set the x-axis limits
# plt.ylim(-15, 15)  # Set the y-axis limits
# plt.legend()
# plt.show()


# # --------------- now COMBINE the filters using convolution -------------------
# # ---------------------- CHECK IF THIS IS CORRECT!! ---------------------------

raw_signal = y

# Design a Bessel low-pass filter:
low_pass_cutoff = 300  # Cutoff frequency in Hz FROM INTAN SETTINGS
low_pass_order = 4    # Filter order FROM INTAN SETTINGS
low_pass_b, low_pass_a = bessel(N=low_pass_order, Wn=low_pass_cutoff / (0.5 * fs), btype='low')

# Design a Bessel high-pass filter:
high_pass_cutoff = 3000   # Cutoff frequency in Hz FROM INTAN SETTINGS
high_pass_order = 4   # Filter order FROM INTAN SETTINGS
high_pass_b, high_pass_a = bessel(N=high_pass_order, Wn=high_pass_cutoff / (0.5 * fs), btype='high')


# Combine the filters via convolution:
combined_b = convolve(low_pass_b, high_pass_b)
combined_a = convolve(low_pass_a, high_pass_a)



# original filter convolution below this:  
combined_b = convolve(low_pass_b, high_pass_b)
combined_a = convolve(low_pass_a, high_pass_a)

# try to modify the convolution since not sure it's working correctly:


# Apply the combined filter to the original signal:
    
DUAL_filtered_signal = lfilter(combined_b, combined_a, raw_signal)


# Plot the original and filtered signals:

plt.figure(figsize=(10, 6))
# plt.plot(time, signal, label='Original Signal')
plt.plot(time, DUAL_filtered_signal, color='red', label='Combined Bessel Filters')
plt.title('Bessel Low-Pass and High-Pass Filters, 110uA Stim, ' + channel_name)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.xlim(2.860, 2.890)  # Set the x-axis limits
plt.ylim(-0.400, 0.400)  # Set the y-axis limits
plt.legend()
plt.show()








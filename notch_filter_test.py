#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:40:07 2024

@author: xander
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.signal import bessel, lfilter, iirnotch, convolve



# ----------------------------------------------------------------------------


os.chdir('/Users/xander/load-rhs-notebook-python/')

from importrhsutilities import *


exec(open("importrhsutilities.py").read())

# file notes: 
    # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
    #  - 20uA

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155351.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)



channel_name = 'D-007' # Change this variable and re-run cell to plot a different channel



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
plt.ylabel("Voltage (mV)")
# plt.xlim(2.885, 2.895)
# plt.ylim(-0.500, 0.500)
plt.title("Raw Amplifier Data")

# ----------------------------------------------------------------------------------------
# below this line try implementation of notch filter - NOT NECESSARY
# ----------------------------------------------------------------------------------------

# signal_data = y

# Define signal parameters
# sampling_rate = 30000  # Hz

# # Define notch filter parameters:
# notch_freq = 60  # Hz
# quality_factor = 30.0  # TRY 30 BUT TEST VALUES AND DISCOVER WHY

# # Design and apply the notch filter
# b, a = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
# filtered_data = signal.filtfilt(b, a, signal_data)

# # Plot the results
# # plt.plot(signal_data, label='Original signal')
# plt.plot(x, filtered_data, label='Filtered signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title("60 Hz Notch Filtered Data")
# plt.xlim(2.880, 2.890)
# plt.ylim(-0.5, 0.5)
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------------
# below this line try thresholding data after notch filtering
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
plt.title('60Hz notch + Thresholding @ ' + str(threshold)+' V')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------
# below this line try bessel bandpass filtering
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
plt.title('110uA Stim, Notch + Threshold + Bessel Bandpass Filter Order = '+str(order))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.xlim(2.840, 2.880)  # Set the x-axis limits
plt.ylim(-0.500, 0.500)  # Set the y-axis limits
plt.legend()
plt.show()

# # ----------------------------------------------------------------------------------------
# # TRY Thresholding BEFORE notch filtering below this line
# # ----------------------------------------------------------------------------------------


# # Apply a threshold to the data as a numpy array:
# threshold = 0.4  # 4 millivolts (???)

# orig_data_arr = np.array(y)

# thresh_signal_arr = np.where(abs(orig_data_arr) > threshold, 0, orig_data_arr)

# thresholded_signal_list = thresh_signal_arr.tolist()

# # OLD thresholding attempt, maybe doesn't work, keep commented: 
# # thresholded_signal_list = [np.sign(sample) * min(np.abs(sample), threshold) for sample in y]

# # Convert the thresholded list back to a NumPy array:
# # thresholded_signal = np.array(thresholded_signal_list)

# # Plot the original and thresholded signals
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='Original Signal')
# plt.plot(x, thresholded_signal_list, label='Thresholded Signal')
# plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
# plt.xlim(2.0, 5.0)
# plt.ylim(-0.500, 0.500)
# plt.title('60Hz notch + Thresholding @ ' + str(threshold)+' V')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.legend()
# plt.show()




# # Define signal parameters
# sampling_rate = 30000  # Hz

# # Define notch filter parameters:
# notch_freq = 60  # Hz
# quality_factor = 30.0  # TRY 30 BUT TEST VALUES AND DISCOVER WHY

# # Design and apply the notch filter
# b, a = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
# filtered_data = signal.filtfilt(b, a, thresholded_signal_list)

# # Plot the results
# # plt.plot(signal_data, label='Original signal')
# plt.plot(x, filtered_data)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title("60 Hz Notch After Thresholding")
# plt.xlim(2.840, 2.880)
# plt.ylim(-0.5, 0.5)
# plt.legend()
# plt.show()



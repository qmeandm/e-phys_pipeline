#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:55:20 2024

@author: xander

Script to find the first stim artifact peak in e-phys data and excludes voltage values in a 
small time window around the peak. 

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
# below this line begin defining data arrays and manipulating
# ----------------------------------------------------------------------------------------

# NOTE: Channel name input is manual for now. You must edit the dictionary position in the
#       rawvolts variable such that Channel D-005 ---> [1]... D-010 --> [6]

# Find the text notes in the INTAN file that give the stim parameters, make a str:
stim_set = str(result['notes']['note1'])
stim_amps = stim_set[:5]

# What channel do you want to analyze data from? INPUT MANUALLY!
nanoclip_channel = 3 # n-1 of the real clip channel! e.g. 0 = acutal channel 1

channel_name = str(nanoclip_channel +1) # set the channel name to call it in the raw data below


# Pull data from the result dictionary in importrhsutilities.py:
    
raw_volts = np.array(result['amplifier_data'][nanoclip_channel]) # Channel D-005 -> [0]... D-010 -> [5]
time = np.array(result['t'])


# ------------------------------------------------------------------------------------------------------
# below this line is a version that finds the first N peaks, excludes stim data, and stores the response
# ------------------------------------------------------------------------------------------------------


def find_peaks(voltage, amplitude_threshold):
    peaks = np.where(voltage > amplitude_threshold)[0]
    return peaks

def extract_data_around_peak(time, voltage, peak_index, num_steps_before, num_steps_after):
    if peak_index == -1:
        return None, None  # Handle the case where no peak was found
    
    start_index = max(0, peak_index - num_steps_before)
    end_index = min(len(time), peak_index + num_steps_after + 1)

    return time[start_index:end_index], voltage[start_index:end_index]



# Set amplitude threshold for peak detection
amplitude_threshold = 200

# Find the first 5 peaks
peaks = find_peaks(raw_volts, amplitude_threshold)[:5]

# Extract and set data around each peak to zero
num_steps_before = 10
num_steps_after = 10

for peak_index in peaks:
    selected_time, selected_voltage = extract_data_around_peak(
        time, raw_volts, peak_index, num_steps_before, num_steps_after
    )

    # Find the indices of selected time steps in the original time array
    original_indices = np.searchsorted(time, selected_time)

    # Set the corresponding original voltage values to zero
    raw_volts[original_indices] = 0

# Plot the original data with selected regions set to zero
plt.plot(time, raw_volts, label='Modified Original Data')
plt.scatter(time[peaks], raw_volts[peaks], color='red', marker='o', label='Selected Peaks')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.xlim(0.1, 5.0)
plt.ylim(-250, 250)
plt.legend()
plt.show()


# ------------------------------------------------------------------------------------------------
# below this line use a version that excludes before/after the peak in regions you manually select
# ------------------------------------------------------------------------------------------------

def find_first_peak(voltage, amplitude_threshold):
    peaks = np.where(voltage > amplitude_threshold)[0]
    if len(peaks) > 0:
        return peaks[0]
    else:
        return -1  # Return a value that indicates no peak was found

def extract_data_around_peak(time, voltage, peak_index, num_steps_before, num_steps_after):
    if peak_index == -1:
        return None, None  # Handle the case where no peak was found
    
    start_index = max(0, peak_index - num_steps_before)
    end_index = min(len(time), peak_index + num_steps_after + 1)

    return time[start_index:end_index], voltage[start_index:end_index]


# Set amplitude threshold for peak detection
amplitude_threshold = 200

# Find the first peak
peak_index = find_first_peak(raw_volts, amplitude_threshold)

if peak_index != -1:
    # Extract data around the peak including an arbitrary number of steps before and after
    num_steps_before = 10
    num_steps_after = 10
    selected_time, selected_voltage = extract_data_around_peak(
        time, raw_volts, peak_index, num_steps_before, num_steps_after
    )


    # Plot original data and selected region
    plt.plot(time, raw_volts, label='Original Data')
    plt.plot(selected_time, selected_voltage, 'r', label='Selected Region')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Peak Excluder 2.0')
    plt.xlim(1.8085, 1.8120)
    # plt.xlim(1.80, 1.85)
    plt.ylim(-800, 800)
    plt.legend()
    plt.show()

# ----------------------------------------------------------------------------------------
# below this line is a version that excludes the peak itself
# ----------------------------------------------------------------------------------------


# os.chdir('/Users/xander/load-rhs-notebook-python/')

# from importrhsutilities import *

# exec(open("importrhsutilities.py").read())

# # file notes: 
#     # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
#     #  - 20uA

# filename = 'RFASDO_ZF_TS Nerve_240201_240201_155222.rhs' # Change this variable to load a different data file
# result, data_present = load_file(filename)


# # NOTE: Channel name input is manual for now. You must edit the dictionary position in the
# # rawvolts variable such that Channel D-005 ---> [1]... D-010 --> [6]

# # Find the text notes in the INTAN file that give the stim parameters, make a str:
# stim_set = str(result['notes']['note1'])
# stim_amps = stim_set[:5]

# # What channel do you want to analyze data from? INPUT MANUALLY!
# nanoclip_channel = 0 # n-1 of the real clip channel! e.g. 0 = acutal channel 1

# channel_name = str(nanoclip_channel +1) # set the channel name to call it in the raw data below


# # Pull data from the result dictionary in importrhsutilities.py:
    
# raw_volts = np.array(result['amplifier_data'][nanoclip_channel]) # Channel D-005 -> [0]... D-010 -> [5]
# time = np.array(result['t'])


# def find_first_peak(voltage, amplitude_threshold):
#     peaks = np.where(voltage > amplitude_threshold)[0]
#     if len(peaks) > 0:
#         return peaks[0]
#     else:
#         return -1  # Return a value that indicates no peak was found

# def extract_data_around_peak(time, voltage, peak_index, num_steps_before, num_steps_after):
#     if peak_index == -1:
#         return None, None  # Handle the case where no peak was found
    
#     start_index = max(0, peak_index - num_steps_before)
#     end_index = min(len(time), peak_index + num_steps_after + 1)
    
#     # Exclude the peak itself
#     if start_index <= peak_index <= end_index:
#         if peak_index - start_index < end_index - peak_index:
#             start_index = peak_index + 1
#         else:
#             end_index = peak_index - 1

#     return time[start_index:end_index], voltage[start_index:end_index]


# # Set amplitude threshold for peak detection
# amplitude_threshold = 200

# # Find the first peak
# peak_index = find_first_peak(raw_volts, amplitude_threshold)

# if peak_index != -1:
#     # Extract data around the peak excluding the peak itself
#     num_steps_before = 0
#     num_steps_after = 150
#     selected_time, selected_voltage = extract_data_around_peak(
#         time, raw_volts, peak_index, num_steps_before, num_steps_after
#     )

#     # Plot original data and selected region
#     plt.plot(time, raw_volts, label='Original Data')
#     plt.plot(selected_time, selected_voltage, 'r', label='Selected Region')
#     # plt.scatter(time[peak_index], raw_volts[peak_index], color='red', marker='o', label='First Peak')
#     plt.xlabel('Time')
#     plt.ylabel('Voltage')
#     plt.title('Peak Excluder')
#     plt.xlim(1.8085, 1.8120)
#     plt.ylim(-800, 800)
#     plt.show()
    















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:42:17 2024

@author: xander
"""

import os
import matplotlib as plt
import numpy as np
from scipy.signal import bessel, lfilter, iirnotch, sosfreqz, sos2tf, convolve
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------


os.chdir('/Users/xander/load-rhs-notebook-python/')

from importrhsutilities import *

exec(open("importrhsutilities.py").read())

# file notes: 
    # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
    #  - 20uA

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155351.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)

# data = result.copy()

# scale_analog_data(result, data)

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
time = np.array(result['t']) # x-axis is the time data

# scaled_data = 0.195 * (raw_volts - 32768) # claim: makes units microvolts


# Plot the raw signal
# plt.plot(time, raw_volts, label='Original Signal', color='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (uV)')
# # plt.xlim(1.805, 1.815) # Use this range for dataset 155222!
# # plt.xlim(1.805, 5.815)
# # plt.ylim(-1500, 1500)
# plt.legend()
# plt.show()


# Make a copy of the original data to manipulate!
modified_raw_volts = raw_volts.copy()

# modified_raw_volts  = scaled_data.copy()

# ------------------------------------------------------------------------------------------------------
# below this line is a version that finds the first N peaks, excludes stim data, and stores the response
# ------------------------------------------------------------------------------------------------------

def extract_data_around_peak(time, voltage, peak_indices, num_steps_before, num_steps_after):
    selected_data = []
    for peak_index in peak_indices:
        start_index = max(0, peak_index - num_steps_before)
        end_index = min(len(time), peak_index + num_steps_after + 1)
        selected_data.append({'time': time[start_index:end_index], 'voltage': voltage[start_index:end_index]})
    return selected_data


# Set amplitude threshold for peak detection
amplitude_threshold = 200

# Find peaks using scipy's find_peaks function
peaks, _ = find_peaks(modified_raw_volts , height=amplitude_threshold)

# Extract and set data around each peak to zero
num_steps_before = 100 # empirically determined
num_steps_after = 5 # empirically determined
max_num_peaks = 10 # find this number of peaks

# Limit to the first 10,000 peaks if available, otherwise use all peaks
selected_peaks = peaks[:min(10000, len(peaks))]

selected_data = extract_data_around_peak(time, modified_raw_volts , selected_peaks, num_steps_before, num_steps_after)


# Set the voltage values corresponding to the selected peaks to zero
peak_counter = 0
for peak_index in selected_peaks:
    start_index = max(0, peak_index - num_steps_before)
    end_index = min(len(time), peak_index + num_steps_after + 1)
    modified_raw_volts[start_index:end_index] = 0
    
    # Increment the counter
    peak_counter += 1
    
    # Check if the desired number of peaks is reached
    if peak_counter >= max_num_peaks:
        break


# Plot the original data with selected regions set to zero
plt.plot(time, modified_raw_volts  , label='Zeroed Stim Pulse Signal')

# plot the regions that have been selected to be zero
# for i, data in enumerate(selected_data):
#     plt.plot(data['time'], data['voltage'], label=f'Peak {i + 1}')

# plt.scatter(time[selected_peaks], raw_volts[selected_peaks], color='red', marker='o', label='Selected Peaks')
plt.plot(time, raw_volts , label='Original Signal', color='red', alpha=0.33)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV?)')
# plt.xlim(1.888, 1.898) # Use this range for dataset 155222!
# plt.xlim(1.805, 9.815)
# plt.ylim(-400, 400)
plt.title('Extracted Peaks (Pink)')
# plt.legend()
plt.show()

''' EVERYTHING ABOVE THIS LINE WORKS, KEEP IT ALL. BELOW IS FOR TESTING '''


# ------------------------------------------------------------------------------------------------------
# below this line run the peak finder for FIRST 10 PEAKS ONLY
# ------------------------------------------------------------------------------------------------------



# # Set amplitude threshold for peak detection
# amplitude_threshold = 200

# # Find peaks using scipy's find_peaks function
# peaks, _ = find_peaks(modified_raw_volts, height=amplitude_threshold)

# # Extract and set data around each peak to zero
# num_steps_before = 100 # empirically determined
# num_steps_after = 5 # empirically determined
# max_num_peaks = 10 # find this number of peaks


# # Limit to the first 10,000 peaks if available, otherwise use all peaks
# selected_peaks = peaks[:min(10000, len(peaks))]


# selected_data = extract_data_around_peak(time, modified_raw_volts, selected_peaks, num_steps_before, num_steps_after)


# peak_counter = 0
# for peak_index in selected_peaks:
#     start_index = max(0, peak_index - num_steps_before)
#     end_index = min(len(time), peak_index + num_steps_after + 1)
#     modified_raw_volts[start_index:end_index] = 0
    
#     # Increment the counter
#     peak_counter += 1
    
#     # Check if the desired number of peaks is reached
#     if peak_counter >= max_num_peaks:
#         break

# # Plot the original data with selected regions set to zero
# plt.plot(time, modified_raw_volts, label='Zeroed Stim Pulse Signal')

# # plot the regions that have been selected to be zero
# # for i, data in enumerate(selected_data):
# #     plt.plot(data['time'], data['voltage'], label=f'Peak {i + 1}')

# # plt.scatter(time[selected_peaks], modified_raw_volts[selected_peaks], color='red', marker='o', label='Selected Peaks')
# plt.plot(time, raw_volts, label='Original Signal', color='red', alpha=0.33)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage')
# plt.xlim(1.805, 1.815) # Use this range for dataset 155222!
# # plt.xlim(1.805, 5.815)
# plt.ylim(-450, 450)
# plt.legend()
# plt.show()


# ------------------------------------------------------------------------------------------------------
# below this line run the peak finder with a band range to detect the responses
# ------------------------------------------------------------------------------------------------------



# def extract_data_around_peak(time, voltage, peak_indices, num_steps_before, num_steps_after):
#     selected_data = []
#     for peak_index in peak_indices:
#         start_index = max(0, peak_index - num_steps_before)
#         end_index = min(len(time), peak_index + num_steps_after + 1)
#         selected_data.append({'time': time[start_index:end_index], 'voltage': voltage[start_index:end_index]})
#     return selected_data

# # Example data generation
# time = np.linspace(0, 10, 1000)
# raw_volts = np.sin(time) + 0.5 * np.random.normal(size=len(time))

# # Set amplitude thresholds for peak detection (above and below)
# threshold_above = 0.8
# threshold_below = -0.8

# # Find peaks using scipy's find_peaks function with both thresholds
# peaks_above, _ = find_peaks(raw_volts, height=threshold_above)
# peaks_below, _ = find_peaks(-raw_volts, height=-threshold_below)

# # Combine the indices of peaks above and below the thresholds
# selected_peaks = np.union1d(peaks_above, peaks_below)

# # Extract and plot data around each selected peak
# num_steps_before = 10
# num_steps_after = 10
# selected_data = extract_data_around_peak(time, raw_volts, selected_peaks, num_steps_before, num_steps_after)

# # Plot the original data with selected regions
# plt.plot(time, raw_volts, label='Original Data')
# for i, data in enumerate(selected_data):
#     plt.plot(data['time'], data['voltage'], label=f'Selected Peak {i + 1}')

# plt.scatter(time[selected_peaks], raw_volts[selected_peaks], color='red', marker='o', label='Selected Peaks')
# plt.axhline(threshold_above, color='green', linestyle='--', label='Threshold Above')
# plt.axhline(-threshold_below, color='blue', linestyle='--', label='Threshold Below')
# plt.xlabel('Time')
# plt.ylabel('Voltage')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

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

#     return time[start_index:end_index], voltage[start_index:end_index]

# # Example data generation
# time = np.linspace(0, 10, 1000)
# raw_volts = np.sin(time) + 0.5 * np.random.normal(size=len(time))

# # Set amplitude threshold for peak detection
# amplitude_threshold = 2

# # Find the first peak
# peak_index = find_first_peak(raw_volts, amplitude_threshold)

# if peak_index != -1:
#     # Extract data around the peak including an arbitrary number of steps before and after
#     num_steps_before = 10
#     num_steps_after = 10
#     selected_time, selected_voltage = extract_data_around_peak(
#         time, raw_volts, peak_index, num_steps_before, num_steps_after
#     )

#     # Find the indices of selected time steps in the original time array
#     original_indices = np.searchsorted(time, selected_time)

#     # Set the corresponding original voltage values to zero
#     raw_volts[original_indices] = 0

#     # Plot the original data with selected region set to zero
#     plt.plot(time, raw_volts, label='Modified Original Data')
#     plt.scatter(selected_time, selected_voltage, color='red', marker='o', label='Selected Region')
#     plt.xlabel('Time')
#     plt.ylabel('Voltage')
#     plt.legend()
#     plt.show()

# else:
#     print("No peak found above the amplitude threshold.")





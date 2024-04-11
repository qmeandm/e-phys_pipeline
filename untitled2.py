#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:20:32 2024

@author: xander

TEST SCRIPT FOR DEBUGGING THE find_peaks FUNCTION - WHY IS MANUAL THRESHOLDING REQ'D ??? TRY USING
prominence SETTING, OR PERHAPS distance SETTING


"""

# ----------------------------------------------------------------------------------------


import os
import matplotlib as plt
import numpy as np
from scipy.signal import bessel, lfilter, iirnotch, sosfreqz, sos2tf, convolve
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------------------


os.chdir('/Users/xander/load-rhs-notebook-python/')
wd = os.getcwd()

from importrhsutilities import *

exec(open("importrhsutilities.py").read())

# file notes: 
    # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
    #  - 20uA

filename = 'RFASDO_ZF_TS Nerve_240201_240201_152522.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)

# channel_name = 'D-009' # Change this variable and re-run cell to plot a different channel

# Leftover code from the Intan python script github for plotting:
# if data_present:
#     plot_channel(channel_name, result)
    
# else:
#     print('Plotting not possible; no data in this file')


# ----------------------------------------------------------------------------------------
#                        begin defining data arrays and manipulating them
# ----------------------------------------------------------------------------------------

# NOTE: Channel name input is manual for now. You must edit the dictionary position in the
#       rawvolts variable such that Channel D-005 ---> [1]... D-010 --> [6]

# Find the text notes in the INTAN file that give the stim parameters, make a str:
stim_set = str(result['notes']['note1'])
stim_amps = stim_set[:5]

# What channel do you want to analyze data from? INPUT MANUALLY! AND CHECK!
nanoclip_channel = 3 # n-1 of the real clip channel! e.g. 0 = acutal channel 1

channel_name = str(nanoclip_channel +1) # set the channel name to call it in the raw data below


# Pull data from the result dictionary in importrhsutilities.py:
    
raw_volts = np.array(result['amplifier_data'][nanoclip_channel]) # Channel D-005 -> [0]... D-010 -> [5]
time = np.array(result['t'])


# keep a copy of the original data for plotting purposes
original_volts = np.array(result['amplifier_data'][nanoclip_channel])

# grab master file notes to use for f string plotting stuff down the line
note1 = result['notes']['note1']
note2 = result['notes']['note2']
note3 = result['notes']['note3']

# ----------------------------------------------------------------------------------------
#                 finding peaks, setting them to zero while preserving ECAPs
# ----------------------------------------------------------------------------------------

def extract_data_around_peak(time, voltage, peak_indices, num_steps_before, num_steps_after):
    selected_data = []
    for peak_index in peak_indices:
        start_index = max(0, peak_index - num_steps_before)
        end_index = min(len(time), peak_index + num_steps_after + 1)
        selected_data.append({'time': time[start_index:end_index], 'voltage': voltage[start_index:end_index]})
    return selected_data


# Set amplitude threshold for peak detection
amplitude_threshold = 30

# Find peaks using scipy's find_peaks function - NEW EDIT - TRY USING PROMINENCE
peaks, _ = find_peaks(raw_volts, height=amplitude_threshold)

# Define number of steps to set to zero 
num_steps_before = 100
num_steps_after = 15

# Limit to the first N peaks if available, otherwise use all peaks
N = 10000
selected_peaks = peaks[:min(N, len(peaks))]

# Curate data using find_peaks and selected_peaks in extract_data_around_peaks
selected_data = extract_data_around_peak(time, raw_volts, selected_peaks, num_steps_before, num_steps_after)


# Set the voltage values corresponding to the selected (thresholding) peaks to zero
for peak_index in selected_peaks:
    start_index = max(0, peak_index - num_steps_before)
    end_index = min(len(time), peak_index + num_steps_after + 1)
    raw_volts[start_index:end_index] = 0


'''
At this point, raw_volts is cleaned of stim artifacts. Now we need to find the
acutal responses, plot them all on the same scale, and plot thew average response.
This is done by using the first difference of the processed signal to find the
regions I set to zero, since we know the response follows directly after
for some characteristic time (<10ms). Using the stored indices of where the zero
signal ends, capture the data in some time window after, store that in a separate
vector, then plot all those vectors + the average of those vectors. This is the 
ECAP we care about.

'''


# Copy the now-edited data for safety and manipulation
responses = raw_volts.copy()

# Vector of first differences so we know where to look for zeroes
first_difference = np.diff(responses)

num_zeros = num_steps_before+num_steps_after # where the zeroes are located
last_zero_indices = []
consecutive_zeros = 0 # set an iterator

# find the set of zeroes and where it ends. append the indices of the zeroes for all data
for i, val in enumerate(first_difference):
  if val == 0:
    consecutive_zeros += 1
  else:
    consecutive_zeros = 0
  if consecutive_zeros == num_zeros:
    last_zero_indices.append(i)
    
# Window size = number of steps before and after the response to save + plot
window_size = 200

window_time = window_size/30 # window size in milliseconds

# Convert sample number to time steps @ 30 kHz sampling
time_list = [step / 30 for step in range(0, window_size)]


# Collect all the ECAPs from the signal in some time interval around 
list_of_windows = []

for i in np.arange(len(last_zero_indices)):
    window = responses[last_zero_indices[i]:last_zero_indices[i]+window_size]
    window = window.reshape(1, window.shape[0])
    list_of_windows.append(window)
    
arr_of_windows = np.concatenate((list_of_windows), axis = 0)


plt.figure()
plt.title(f"All ECAPs, {stim_amps} Stim Current")
# plt.suptitle(f'Window Size: {window_size} steps')
for i in np.arange(len(list_of_windows)):
    dat = list_of_windows[i].T
    plt.plot(time_list, dat, label = f'Response {i}')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage ($\mu$V)")
    

# plt.legend()
plt.show()

# Calculate the average response of all those found and plot it seperately
mean_response = np.mean(arr_of_windows, axis = 0)

plt.figure()
plt.title(f"Average ECAP, {stim_amps} Stim Current")
plt.plot(time_list, mean_response)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage ($\mu$V)")
plt.show()

num_traces_plotted = len(arr_of_windows)

print(f'Number of traces plotted = {num_traces_plotted}')
print(f'Note 1 = {note1}')
print(f'Note 2 = {note2}')
print(f'Note 3 = {note3}')
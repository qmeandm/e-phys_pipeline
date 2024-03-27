#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:20:32 2024

@author: xander

UPDATED: 3/25/2024

Outstanding Issues:
    - ? ? ? ? ? ? ? ? ? ? ? IS THE VOLTAGE SCALE CORRECT? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? 
    - right now the stim pulse amplitude threshold is arbitrary and MANUAL for each recording!
    - so are the number of steps before + after the peak - dangerous for different recordings!
    - plotting the time windows for each peak will vary across all data...
    - why does the thresholding have different values across peaks? it should be constant...

"""

# ----------------------------------------------------------------------------------------


import os
import matplotlib as plt
import numpy as np
from scipy.signal import bessel, lfilter, iirnotch, sosfreqz, sos2tf, convolve
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------------------


os.chdir('/Users/xander/load-rhs-notebook-python/')

from importrhsutilities import *

exec(open("importrhsutilities.py").read())

# file notes: 
    # 155351 - 120uA , 155256 - 110 uA, 155222 - 100 uA, 162231 - 40 uA post washout
    #  - 20uA

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155222.rhs' # Change this variable to load a different data file
result, data_present = load_file(filename)


# ----------------------------------------------------------------------------------------
#                        begin defining data arrays and manipulating them
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


# keep a copy of the original data for plotting purposes
original_volts = np.array(result['amplifier_data'][nanoclip_channel])

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
amplitude_threshold = 200

# Find peaks using scipy's find_peaks function
peaks, _ = find_peaks(raw_volts, height=amplitude_threshold)

# Define number of steps to set to zero 
num_steps_before = 100
num_steps_after = 5

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


# Plot the original data with selected regions set to zero
plt.figure()
plt.plot(time, raw_volts, label='Original Data')
plt.show()

plt.figure()
for i, data in enumerate(selected_data):
    plt.plot(data['time'], data['voltage'], label=f'Peak {i + 1}')

plt.figure()
plt.scatter(time[selected_peaks], raw_volts[selected_peaks], color='red', marker='o', label='Selected Peaks')
plt.plot(time, original_volts, label='Original Signal', color='green', alpha=0.33)
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Example of Stim Artifact Removal, Single Channel, 120 uA')
plt.xlim(1.805, 1.815)
plt.ylim(-250, 250)
plt.show()


# Plot to show that the raw_volts signal has been correctly cleaned of stim artifact
# plt.plot(time, raw_volts, label='Zeroed Signal', color='purple')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage')
# plt.xlim(1.805, 1.815)
# plt.ylim(-250, 250)
# plt.show()


# ----------------------------------------------------------------------------------------
#                         find the ECAP peaks, plot them all
# ----------------------------------------------------------------------------------------


# At this point, raw_volts is now cleaned of stim artifacts. Now we need to find the first
# N peaks, set their x-values to an arbitrary timescale, average them, and plot that value
# as a single peak. Also plot all N peaks overlaid for variation visualization

# Copy the now-edited data for second round of thresholding
responses = raw_volts.copy()

# Set a new amplitude threshold for peak detection
amplitude_threshold = 50

# # Define number of steps to set to zero 
# num_steps_before = 100
# num_steps_after = 100

# The following code will use the first difference of the processed signal to find the ROI for averaging (down below)

first_difference = np.diff(responses)

num_zeros = num_steps_before+num_steps_after
last_zero_indices = []
consecutive_zeros = 0
for i, val in enumerate(first_difference):
  if val == 0:
    consecutive_zeros += 1
  else:
    consecutive_zeros = 0
  if consecutive_zeros == num_zeros:
    last_zero_indices.append(i)
    
window_size = 5000

list_of_windows = []

for i in np.arange(len(last_zero_indices)):
    window = responses[last_zero_indices[i]:last_zero_indices[i]+window_size]
    window = window.reshape(1, window.shape[0])
    list_of_windows.append(window)
    
arr_of_windows = np.concatenate((list_of_windows), axis = 0)


plt.figure()
plt.suptitle(f"Plot of Stimulus Response")
plt.title(f'Window Size: {window_size}')
for i in np.arange(len(list_of_windows)):
    dat = list_of_windows[i].T
    plt.plot(dat, label = f'Response {i}')

plt.legend()
plt.show()


mean_response = np.mean(arr_of_windows, axis = 0)

plt.figure()
plt.title("Average Response")
plt.plot(mean_response)
plt.xlabel("Sample Number")
plt.ylabel("Voltage (a.u.)")
plt.show()















# Find new peaks using scipy's find_peaks function
peaks2, _ = find_peaks(responses, height=amplitude_threshold)

# Limit to the first N peaks if available, otherwise use all peaks
N = 10000
selected_peaks2 = peaks2[:min(N, len(peaks))]

# Create an empty list to store peak data
peak_data = [] 

# Loop through the indices of the found peaks to extract the data and store in peak_data vector
for peak_index in selected_peaks2:
  start_index = max(0, peak_index - num_steps_before)
  end_index = min(len(time), peak_index + num_steps_after + 1)

  # Extract the data from responses
  peak_segment = responses[start_index:end_index]

  # Append the extracted data to the peak_data list
  peak_data.append(peak_segment)


plt.figure()
for i, peak_segment in enumerate(peak_data):
    # Label each plot with a unique name
    label = f"Peak {i+1}"
    
    plt.plot(peak_segment, label = label)
    
plt.legend()
plt.show()







# The below code to plot the multiple peaks works good, but the traces don't share x-axis values
fig, ax = plt.subplots()  # Create a figure and an axes object

for i, peak_segment in enumerate(peak_data):
    # Label each plot with a unique name
    label = f"Peak {i+1}"

    # Plot the peak segment on the same axes
    ax.plot(time[start_index:end_index], peak_segment, label=label)

# Add labels and a legend
ax.set_xlabel("Time")
ax.set_ylabel("Voltage")
plt.tight_layout()
# ax.legend()
plt.title('All ECAPs Found, Single Channel, 120 uA')
plt.show()



# Plot the average value of all the discovered response spikes

# Calculate average voltage directly using time and peak_data
total_voltage = np.zeros_like(time)  # Initialize accumulator for voltage sum
num_segments = np.zeros_like(time)  # Counter for number of segments contributing to each time point

for peak_segment in peak_data:
    peak_times = time[peak_index - num_steps_before : peak_index + num_steps_after + 1]  # Extract time points for this segment
    
    # Convert peak_times to integer indices
    integer_indices = np.searchsorted(time, peak_times)

    total_voltage[integer_indices] += peak_segment
    num_segments[integer_indices] += 1

average_voltage = total_voltage / num_segments

# Plot average voltage
fig, ax = plt.subplots()  # Create a figure and an axes object

ax.plot(time, average_voltage, label="Average Voltage")

# Add labels, legend, and title
ax.set_xlabel("Time")
ax.set_ylabel("Voltage")
plt.tight_layout()
# ax.legend()
plt.title('Average of All ECAPs, Single Channel, 120 uA')
plt.show()

print(max(average_voltage))





# ------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# def find_peaks(voltage, amplitude_threshold):
#     peaks = np.where(voltage > amplitude_threshold)[0]
#     return peaks

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
# amplitude_threshold = 1.5

# # Find the first 5 peaks
# peaks = find_peaks(raw_volts, amplitude_threshold)[:5]

# # Extract and set data around each peak to zero
# num_steps_before = 10
# num_steps_after = 10

# for peak_index in peaks:
#     selected_time, selected_voltage = extract_data_around_peak(
#         time, raw_volts, peak_index, num_steps_before, num_steps_after
#     )

#     # Find the indices of selected time steps in the original time array
#     original_indices = np.searchsorted(time, selected_time)

#     # Set the corresponding original voltage values to zero
#     raw_volts[original_indices] = 0

# # Plot the original data with selected regions set to zero
# plt.plot(time, raw_volts, label='Modified Original Data')
# plt.scatter(time[peaks], raw_volts[peaks], color='red', marker='o', label='Selected Peaks')
# plt.xlabel('Time')
# plt.ylabel('Voltage')
# plt.xlim(0.1, 1.5)
# plt.ylim(-2, 2)
# plt.legend()
# plt.show()

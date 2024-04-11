#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:20:32 2024

@author: xander

ISSUES UPDATED: 4/10/2024

Outstanding Issues:
    - right now the stim pulse amplitude threshold is arbitrary and MANUAL for each recording!
    - so are the number of steps before + after the peak - dangerous for different recordings?
    - why does the thresholding have different values across peaks? it should be constant...
        - this really is an issue with find_peaks that needs to be addressed

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

filename = 'RFASDO_ZF_TS Nerve_240201_240201_155222.rhs' # Change this variable to load a different data file
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
amplitude_threshold = 100

# Find peaks using scipy's find_peaks function
peaks, _ = find_peaks(raw_volts, height=amplitude_threshold)

# Define number of steps to set to zero 
num_steps_before = 100
num_steps_after = 15

# Limit to the first N peaks if available, otherwise use all peaks
N = 120
selected_peaks = peaks[:min(N, len(peaks))]

# tell me how many peaks the peak finder found
number_of_peaks_found = len(selected_peaks)
print(f"Number of peaks found: {number_of_peaks_found}")

# Curate data using find_peaks and selected_peaks in extract_data_around_peaks
selected_data = extract_data_around_peak(time, raw_volts, selected_peaks, num_steps_before, num_steps_after)


# Plot the original data
plt.plot(raw_volts)

# Mark the peaks with red circles
plt.plot(peaks, raw_volts[peaks], 'ro', markersize=8, label='Peaks')  # Adjust markersize as needed

# Customize the plot for clarity
plt.xlabel('Time (or other unit)')
plt.ylabel('Voltage')
plt.title('Peaks in Raw Voltage Data')
plt.legend()
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()



# Set the voltage values corresponding to the selected (thresholding) peaks to zero
for peak_index in selected_peaks:
    start_index = max(0, peak_index - num_steps_before)
    end_index = min(len(time), peak_index + num_steps_after + 1)
    raw_volts[start_index:end_index] = 0


# Plot the original data with selected regions set to zero
# plt.figure()
# plt.plot(time, raw_volts, label='Original Data')
# plt.show()

# plt.figure()
# for i, data in enumerate(selected_data):
#     plt.plot(data['time'], data['voltage'], label=f'Peak {i + 1}')

# plt.figure()
# plt.scatter(time[selected_peaks], raw_volts[selected_peaks], color='red', marker='o', label='Selected Peaks')
# plt.plot(time, original_volts, label='Original Signal', color='green', alpha=0.33)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage')
# plt.title(f'Example of Stim Artifact Removal, Single Channel, {stim_amps} ')
# plt.xlim(1.805, 1.815) #this range only valid for dataset 155222
# plt.ylim(-250, 250) # this range obly valid for dataset 155222
# plt.show()


# Plot to show that the raw_volts signal has been correctly cleaned of stim artifact
# plt.plot(time, raw_volts, label='Zeroed Signal', color='purple')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage')
# plt.xlim(1.805, 1.815)
# plt.ylim(-250, 250)
# plt.show()


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

'''
Below I'm showing that there is no significant difference in the peak ECAP value
across channels in the nanoclip - this suggests that the location of the electrode does
not factor into our e-phys results, and that we are justified in choosing a single 
channel to plot ECAPs from
'''


maxes = []
for i in range(len(arr_of_windows)):
    values = arr_of_windows[i]
    maxes.append(max(values[values > 0]))
    
# Calculate pairwise distances using a Euclidean distance function:
def pairwise_distance(p1, p2):
  """Calculates the Euclidean distance between two points."""
  return np.sqrt(np.sum((p1 - p2) ** 2))

data = maxes

# Calculate pairwise distances in maxes using np.subtract.outer
pairwise_distances = np.subtract.outer(data, data)

# Flatten the pairwise distances array for plotting the distribution
flat_distances = pairwise_distances.flatten()

# Plot the distribution of the pairwise distances
plt.hist(flat_distances)
plt.xlabel("Pairwise Distance")
plt.ylabel("Frequency")
plt.title(f"Distribution of Pairwise Distance between Maximum ECAP, {stim_amps} Stim, Channel {nanoclip_channel}")
plt.show()

# Calculate descriptive statistics
mean_distance = np.mean(flat_distances)
median_distance = np.median(flat_distances)
std_distance = np.std(flat_distances)

print(f"Mean Distance: {mean_distance:.5f}")
print(f"Median Distance: {median_distance:.5f}")
print(f"Standard Deviation: {std_distance:.5f}")


''' 
Below I'm doing the pairwise calculation and plotting for the maxes of all 6
channels
'''

# define lists to dump descriptive stats into

pairwise_means = []
pairwise_sds = []

for k in range(6):
    stim_set = str(result['notes']['note1'])
    stim_amps = stim_set[:5]

    # What channel do you want to analyze data from? INPUT MANUALLY! AND CHECK!
    nanoclip_channel = k # n-1 of the real clip channel! e.g. 0 = acutal channel 1

    channel_name = str(nanoclip_channel +1) # set the channel name to call it in the raw data below


    # Pull data from the result dictionary in importrhsutilities.py:
        
    raw_volts = np.array(result['amplifier_data'][nanoclip_channel]) # Channel D-005 -> [0]... D-010 -> [5]
    time = np.array(result['t'])


    # keep a copy of the original data for plotting purposes
    original_volts = np.array(result['amplifier_data'][nanoclip_channel])
    
        
    # Set amplitude threshold for peak detection
    amplitude_threshold = 100
    
    # Find peaks using scipy's find_peaks function
    peaks, _ = find_peaks(raw_volts, height=amplitude_threshold)
    
    # Define number of steps to set to zero 
    num_steps_before = 100
    num_steps_after = 15
    
    # Limit to the first N peaks if available, otherwise use all peaks
    N = 120
    selected_peaks = peaks[:min(N, len(peaks))]
    
    
    # Curate data using find_peaks and selected_peaks in extract_data_around_peaks
    selected_data = extract_data_around_peak(time, raw_volts, selected_peaks, num_steps_before, num_steps_after)
    
    
    
    # Set the voltage values corresponding to the selected (thresholding) peaks to zero
    for peak_index in selected_peaks:
        start_index = max(0, peak_index - num_steps_before)
        end_index = min(len(time), peak_index + num_steps_after + 1)
        raw_volts[start_index:end_index] = 0

        
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

    maxes = []
    for i in range(len(arr_of_windows)):
        values = arr_of_windows[i]
        maxes.append(max(values[values > 0]))

    data = maxes

    # Calculate pairwise distances in maxes using np.subtract.outer
    pairwise_distances = np.subtract.outer(data, data)

    # Flatten the pairwise distances array for plotting the distribution
    flat_distances = pairwise_distances.flatten()

    # Plot the distribution of the pairwise distances
    plt.hist(flat_distances, alpha=0.5, label=channel_name)
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Pairwise Distance between maximum ECAPs, {stim_amps} Stim, All Channels")
    plt.legend()
    plt.show()

    # Calculate descriptive statistics
    mean_distance = np.mean(flat_distances)
    median_distance = np.median(flat_distances)
    std_distance = np.std(flat_distances)
    
    pairwise_means.append(mean_distance)
    pairwise_sds.append(std_distance)

    # print(f"Mean Distance: {mean_distance:.5f}")
    # print(f"Median Distance: {median_distance:.5f}")
    # print(f"Standard Deviation: {std_distance:.5f}")

print("Pairwise Means = " + str(pairwise_means))
# print(f"Median Distance: {median_distance:.5f}")
# print(f"Standard Deviation: {pairwise_sds:.5f}")
print("Pairwise SDs = " + str(pairwise_sds))


# ----------------------------------------------------------------------------------------
# below here I'm trying to automate plotting all the data with the caveat that some data needs
# different peak detector thresholding. currently the loop just prompts the user for the
# threshold before plotting everything. this would be better if there was no need to
# manually update the threshold...
# ----------------------------------------------------------------------------------------



# # Define the folder path containing your datasets
# data_folder = "/Users/xander/load-rhs-notebook-python/plot_data"

# # Find all files in the folder
# data_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

# # Minimum and maximum y-values to ensure consistent scaling
# min_y = float('inf')
# max_y = float('-inf')

# # Loop through each data file
# for filename in data_files:
#   # Load the data (replace 'load_data' with your specific function)
#   x, y = load_data(os.path.join(data_folder, filename))

#   # Print filename and prompt user for amplitude threshold (with loop)
#   while True:
#     print(f"Plotting data from: {filename}")
#     try:
#       amplitude_threshold = float(input("Enter amplitude threshold (numerical value): "))
#       break  # Exit the loop if valid input is received
#     except ValueError:
#       print("Invalid input. Please enter a numerical value.")

#   # Update min and max values
#   min_y = min(min_y, min(y))
#   max_y = max(max_y, max(y))

 

#   # Plot the data with a unique label
#   plt.plot(x, y, label=filename)

# # Set the y-axis limits for consistent scaling
# plt.ylim(min_y, max_y)

# # Add labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("All Datasets")

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()

# # Function to load data (replace with your actual loading logic)
# def load_data(filepath):
#   # Implement your logic to read x and y data from the file
#   # ... (same as before)






















# create temporary empty dict for manually plotting relevant data
responses_dict = {}  # first time you run the code only!

# Append the arrays as separate entries with meaningful keys
responses_dict["time"] = time_list
responses_dict["response 110uA"] = mean_response

# Accessing data for plotting
# Assuming you're using Matplotlib
import matplotlib.pyplot as plt

# Access data by key
t_data = responses_dict["time"]
current_response = responses_dict["response 110uA"]


# Plot the data
# plt.figure()
# plt.title("Average ECAP, 10 - 120 uA Stim Current")
# plt.plot(t_data, current_response)
# plt.xlabel("Time (ms)")
# plt.ylabel("Voltage ($\mu$V)")
# plt.show()












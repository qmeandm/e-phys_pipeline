#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:03:12 2024

@author: xander

Testing a file finder loop for Intan Reader script so we can loop through all the files
in a directory/sub directory, and read the .rhs files. Evenrually this will be the
wrapper for the AS_intan_reader script!

"""

import os

def process_files(directory_path, file_extension):
    # Define the working directory
    working_directory = os.getcwd()
    print("Current working directory:", working_directory)

    try:
        # Change to the specified directory for loading files
        os.chdir(directory_path)
        print("Changed to loading directory:", directory_path)

        # Loop through all files in the loading directory with the given file extension
        for filename in os.listdir(directory_path):
            if filename.endswith(file_extension):
                file_path = os.path.join(directory_path, filename)
                print("Processing file:", file_path)

                # Example operation: Print the contents of each text file
                if file_extension == '.txt':
                    with open(file_path, 'r') as file:
                        file_contents = file.read()
                        print("File contents:")
                        print(file_contents)

                # Add your file processing logic here
                # For example, you can perform any desired operations on the file

    except FileNotFoundError:
        print("Loading directory not found:", directory_path)
    finally:
        # Change back to the original working directory
        os.chdir(working_directory)
        print("Changed back to original working directory:", working_directory)

# Example usage
loading_directory = '/Users/xander/Documents/Experiments/20240201 ZF TS Nerve Smania Device - E-Phys Data /test dir for scripting'  # Replace with the path of the directory to load files from
file_extension = '.txt'  # Replace with the desired file extension
process_files(loading_directory, file_extension)



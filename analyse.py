import os
import glob

# Specify the directory containing the log files
logfiles_dir = 'results/20240713_da7c40f/'

# Pattern to match all files
pattern = os.path.join(logfiles_dir, '*')

# Use glob to find files matching the pattern
all_file_paths = glob.glob(pattern)

# Exclude .pkl files
logfile_paths = [path for path in all_file_paths if not path.endswith('.pkl')]

# List for storing paths of log files with Python errors
error_logfiles = []

# Check each file for Python errors
for logfile_path in logfile_paths:
    with open(logfile_path, 'r') as file:
        contents = file.read()
        # Check if the file contains a Python error signature
        if 'Trace' in contents or 'error' in contents:
            error_logfiles.append(logfile_path)

# Print or process the log files with Python errors
for i, error_logfile in enumerate(error_logfiles):
    print(f"{i+1} {error_logfile}")
    # print first line of the log file
    with open(error_logfile, 'r') as file:
        print(file.readline())

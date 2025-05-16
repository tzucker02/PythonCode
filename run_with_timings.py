# run_with_timings.py
#
#!/usr/bin/env python
# coding: utf-8

dict_def = input("For first time use, or to clean up, YES to initialize the timing output dictionary, NO to restore dictionary from previous runs: ")

# check input above

if dict_def.upper() == "YES":
    time_dict = {}
elif dict_def.upper() == "Y":
    time_dict = {}
elif dict_def.upper() == "NO":
    # don't initialize the time dictionary, but recall any stored values
    get_ipython().run_line_magic('store', '-r')
elif dict_def.upper() == "N":
    # don't initialize the time dictionary, but recall any stored values
    get_ipython().run_line_magic('store', '-r')

import time
import subprocess
import sys

start_time = time.time()
file_to_run = input("Full path to file (without ipynb extension - e.g., myjupyternotebook): ")
file_type = input("Is this a jupyter notebook?: ")

# check file-type entered above

if file_type.upper() == "YES":
    file_to_run = file_to_run + ".ipynb"
elif file_type.upper() == "Y":
    file_to_run = file_to_run + ".ipynb"
else:
    print("Not a Jupyter Notebook.")

# Use subprocess to run the external Python file instead of %run
try:
    subprocess.run([sys.executable, file_to_run], check=True)
except Exception as e:
    print(f"Error running file: {e}")

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_inminutes = elapsed_time/60
print(f"Total run-time: \033[1m{elapsed_time:.4f} seconds\033[0m or \033[1m{elapsed_time_inminutes:.4f} minutes\033[0m")
print(" ")

import datetime
now = datetime.datetime.now()
timenow = now.strftime("%Y-%m-%d_%H:%M:%S")
now = timenow
# bold variables before making them a dictionary key
ctime = f"\033[1m{now}\033[0m"
file_to_run = f"\033[1m{file_to_run}\033[0m"
keyname = "File Name: " + file_to_run + ", " + "Date and Time of run: " + ctime
time_dict.update({keyname:elapsed_time})
get_ipython().run_line_magic('store', 'time_dict')

for key, value in time_dict.items():
    value_in_min = int(value/60)
    value_seconds = value % 60
    print(f"{key}\tTime to run notebook:\t\033[1m{value_in_min:.0f}\033[0m minutes and \033[1m{value_seconds:06.3f}\033[0m seconds")
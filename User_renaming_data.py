import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
from datetime import time
import re


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Raw Data\P6\Force data'
ID = os.path.basename(directory_path)
folder = directory_path
all_files = os.listdir(directory_path)
time_pattern = re.compile(r"(\d\d)_(\d\d)_(\d\d)\.csv$")
directory_path_after_renaming = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Force data'

files_with_time = []
for filename in all_files:
    match = time_pattern.search(filename)
    if match:
        hh = int(match.group(1))
        mm = int(match.group(2))
        ss = int(match.group(3))
        t = time(hh, mm, ss)
        files_with_time.append((filename, t))
        print(files_with_time)

files_with_time.sort(key=lambda x: x[1])
print("Chronological order:")
for number, (fname, t) in enumerate(files_with_time, start=1):
    print(number, fname, t)

for number, (old_name, t) in enumerate(files_with_time, start=1):
    if number != 10:
        new_name = f"Training_set_{number}.csv"
        os.rename(os.path.join(directory_path, old_name),
                  os.path.join(directory_path_after_renaming, new_name))
        print(f"Renamed {old_name} → {new_name}")
    else:
        new_name = f"Training_set_with_pert.csv"
        os.rename(os.path.join(directory_path, old_name),
                  os.path.join(directory_path_after_renaming, new_name))
        print(f"Renamed {old_name} → {new_name}")

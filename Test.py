import os

import numpy as np
import matplotlib.pyplot as plt
import lib
import pandas as pd
import Lib_grip as lb
import polars as pl


directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals\Collection\White_1'
for file in os.listdir(directory):
    for i in range(1, 11):
        file_path = os.path.join(directory, f"T{i}.txt")
        signal = lb.read_my_txt_file(file_path)
        print(np.average(signal))
        print(np.std(signal))
        plt.plot(signal)
        plt.show()


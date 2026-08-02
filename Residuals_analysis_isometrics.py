import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from Lib_grip import spatial_error
import glob
import lib



directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen'

files = glob.glob(os.path.join(directory_path, "*"))



list_ID = []
list_ID_team = []
Isometric_high_fc = []
Isometric_low_fc = []

for file in files:
    ID = os.path.basename(file)
    print(ID) # We keep this so that we know which participant is assessed during the run of the code
    list_ID.append(ID)
    ID_team = ID.split("_")
    list_ID_team.append(ID_team[0])

    directory = file + r"\Grip data"
    os.chdir(directory)
    Isometric_high = pd.read_csv("Isometric_high.csv", skiprows=2)
    Isometric_low = pd.read_csv("Isometric_low.csv", skiprows=2)
    Isometric_high_force_output = Isometric_high['Performance'].to_numpy()
    Isometric_low_force_output = Isometric_low['Performance'].to_numpy()

    fs = 100
    cutoff_freqs = np.linspace(1,40, 40)
    name_high = ID + "_residual_Isometric_high.png"
    name_low = ID + "_residual_Isometric_low.png"

    directory_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results\Residual analysis\Isometric data'
    fc_high = lib.Residual_analysis(var=Isometric_high_force_output, cutoff_frequencies=cutoff_freqs, fs=fs, number_of_fit_points=10, plot=False, save=(name_high, directory_save))
    fc_low = lib.Residual_analysis(var=Isometric_low_force_output, cutoff_frequencies=cutoff_freqs, fs=fs, number_of_fit_points=10, plot=False, save=(name_low, directory_save))

    Isometric_high_fc.append(fc_high)
    Isometric_low_fc.append(fc_low)

plt.plot(list_ID, Isometric_high_fc, label="Isometric_high_fc")
plt.plot(list_ID, Isometric_low_fc, label="Isometric_low_fc")
plt.legend()
plt.show()






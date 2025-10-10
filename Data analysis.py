import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib
import Lib_grip as lb

force_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Raw Data\Grip assessment 10_9_2025\K-Grip'
os.chdir(force_directory)
data_force = pd.read_csv(r'grip_strength_Damianou__Anestis___09Oct25_12_46_52.csv', skiprows=2)


artinis_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Raw Data\Grip assessment 10_9_2025\Artinis'
name = r'Sine_P_1'
data, sampling_frequency = lb.artinis_read_file(artinis_directory, name)


idx = data.index[data["(Event)"] == "A1 "].to_list()
trial_range_start = idx[0]
trial_range_stop = trial_range_start + (sampling_frequency * 20)
print()
data_trial = data.loc[trial_range_start : trial_range_stop]

brain_signal = lib.Butterworth_highpass(sampling_frequency, 0.05, data_trial['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'])
print(data_trial.columns)
plt.plot(data_trial['Time'], data_trial['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'], label='Original')
plt.plot(data_trial['Time'], brain_signal, label='Filtered')
plt.legend()
plt.show()

print(f"DFA = {lb.dfa(data_trial['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'], plot=True)}")
print(f"SaEn = {lb.Ent_Samp(data_trial['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'], 2, 0.2)}")
print(f"Frequencies = {lib.FFT(data_trial['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'], sampling_frequency)}")

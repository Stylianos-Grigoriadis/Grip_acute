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
name = r'White_P_1'
data, sampling_frequency = lb.artinis_read_file(artinis_directory, name)


idx = data.index[data["Event"] == "A1 "].to_list()
trial_range_start = idx[0]
trial_range_stop = trial_range_start + (sampling_frequency * 30)
print()
data_trial = data.loc[trial_range_start : trial_range_stop]
print(data_trial.columns)
Rx1_Tx1_O2Hb = data_trial['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
Rx1_Tx2_O2Hb = data_trial['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
Rx1_Tx3_O2Hb = data_trial['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
Rx2_Tx1_O2Hb = data_trial['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
Rx2_Tx2_O2Hb = data_trial['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
Rx2_Tx3_O2Hb = data_trial['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
list_data = [Rx1_Tx1_O2Hb, Rx1_Tx2_O2Hb, Rx1_Tx3_O2Hb, Rx2_Tx1_O2Hb, Rx2_Tx2_O2Hb, Rx2_Tx3_O2Hb]
list_data_names = ['Rx1_Tx1_O2Hb', 'Rx1_Tx2_O2Hb', 'Rx1_Tx3_O2Hb', 'Rx2_Tx1_O2Hb', 'Rx2_Tx2_O2Hb', 'Rx2_Tx3_O2Hb']

# Rx1_Tx2_O2Hb_highpass = lib.Butterworth_highpass(sampling_frequency, 0.05, Rx1_Tx2_O2Hb)
# Rx1_Tx3_O2Hb_highpass = lib.Butterworth_highpass(sampling_frequency, 0.05, Rx1_Tx3_O2Hb)
# Rx2_Tx1_O2Hb_highpass = lib.Butterworth_highpass(sampling_frequency, 0.05, Rx2_Tx1_O2Hb)
# Rx2_Tx2_O2Hb_highpass = lib.Butterworth_highpass(sampling_frequency, 0.05, Rx2_Tx2_O2Hb)
# Rx2_Tx3_O2Hb_highpass = lib.Butterworth_highpass(sampling_frequency, 0.05, Rx2_Tx3_O2Hb)
# for name, data in zip(list_data_names, list_data):
#     filtered = lib.Butterworth_band(sampling_frequency, [0.01, 0.2], data)
#     plt.plot(data_trial['Time'], filtered, label='Filtered')
#     plt.plot(data_trial['Time'], data, label='Original')
#
#     plt.title(name)
#     plt.legend()
#     plt.show()

# print(f"DFA = {lb.dfa(Rx1_Tx2_O2Hb_highpass, plot=True)}")
# print(f"SaEn = {lb.Ent_Samp(Rx1_Tx2_O2Hb_highpass, 2, 0.2)}")
# print(f"Frequencies = {lib.FFT(Rx1_Tx2_O2Hb_highpass, sampling_frequency)}")


# lb.fNIRS_check_quality(Rx1_Tx1_O2Hb, 100, plot=True)
# lb.fNIRS_check_quality(Rx1_Tx2_O2Hb, 100, plot=True)
# lb.fNIRS_check_quality(Rx1_Tx3_O2Hb, 100, plot=True)
# lb.fNIRS_check_quality(Rx2_Tx1_O2Hb, 100, plot=True)
# lb.fNIRS_check_quality(Rx2_Tx2_O2Hb, 100, plot=True)
# lb.fNIRS_check_quality(Rx2_Tx3_O2Hb, 100, plot=True)
# plt.plot(Rx1_Tx1_O2Hb, label='Rx1_Tx1_O2Hb')
# plt.plot(Rx1_Tx2_O2Hb, label='Rx1_Tx2_O2Hb')
# plt.plot(Rx1_Tx3_O2Hb, label='Rx1_Tx3_O2Hb')
# plt.plot(Rx2_Tx1_O2Hb, label='Rx2_Tx1_O2Hb')
# plt.plot(Rx2_Tx2_O2Hb, label='Rx2_Tx2_O2Hb')
# plt.plot(Rx2_Tx3_O2Hb, label='Rx2_Tx3_O2Hb')
# plt.legend()
# plt.show()
fs = 100
for signal in list_data:
    original = signal.copy()
    mask, z = lb.detect_motion_mask_from_movstd(2, signal, fs)
    segs = lb.mask_to_segments(mask, len(signal), fs, z, signal, thresh_z=4, plot=True)
    if segs:
        repair_interp = lb.repair_motion_linear(signal, segs, fs)
        repair_spline = lb.repair_motion_scholkmann(signal, segs, fs)
        original_filtered = lb.butter_bandpass_filtfilt(original, fs, low=0.01, high=0.30)
        repair_interp_filtered = lb.butter_bandpass_filtfilt(repair_interp, fs, low=0.01, high=0.30)
        repair_spline_filtered = lb.butter_bandpass_filtfilt(repair_spline, fs, low=0.01, high=0.30)

        plt.plot(repair_interp, label='repair_interp')
        plt.plot(repair_spline, label='repair_spline')
        plt.plot(original, label='original')
        plt.plot(repair_interp_filtered, label='repair_interp_filtered')
        plt.plot(repair_spline_filtered, label='repair_spline_filtered')
        plt.plot(original_filtered, label='original_filtered')

        plt.legend()
        plt.show()











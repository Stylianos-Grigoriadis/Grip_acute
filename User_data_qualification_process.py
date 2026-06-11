import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os


#     # The Rx1 and Rx3 are the long distance receiver
#     # The 9322 is left side and the 9323 is right side

plt.rcParams['font.family'] = 'serif'        # e.g., 'serif', 'sans-serif', 'monospace'
plt.rcParams['font.size'] = 16
directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen\Sine_1'
os.chdir(directory)

parts = directory.split(os.sep)
ID = parts[-1]
print(ID)

grip_directory = directory + r'\Grip data'
os.chdir(grip_directory)
training_set_1 = pd.read_csv(r'Training_1.csv', skiprows=2)
training_set_2 = pd.read_csv(r'Training_2.csv', skiprows=2)
training_set_3 = pd.read_csv(r'Training_3.csv', skiprows=2)
training_set_4 = pd.read_csv(r'Training_4.csv', skiprows=2)
training_set_5 = pd.read_csv(r'Training_5.csv', skiprows=2)
training_set_6 = pd.read_csv(r'Training_6.csv', skiprows=2)
training_set_7 = pd.read_csv(r'Training_7.csv', skiprows=2)
training_set_8 = pd.read_csv(r'Training_8.csv', skiprows=2)
training_set_9 = pd.read_csv(r'Training_9.csv', skiprows=2)
training_set_10 = pd.read_csv(r'Training_10.csv', skiprows=2)


training_set_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_1)
training_set_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_2)
training_set_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_3)
training_set_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_4)
training_set_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_5)
training_set_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_6)
training_set_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_7)
training_set_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_8)
training_set_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_9)
training_set_10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_10)

list_set = [training_set_1,
            training_set_2,
            training_set_3,
            training_set_4,
            training_set_5,
            training_set_6,
            training_set_7,
            training_set_8,
            training_set_9,
            training_set_10
            ]

fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
axes = axes.flatten()  # Make indexing easier

for i, ax in enumerate(axes):
    set = list_set[i]

    ax.plot(set['Time'], set['Performance'], label='Force Output')
    ax.plot(set['ClosestSampleTime'], set['Target'], label='Target')

    ax.set_title(f"Set {i + 1}")
    ax.legend(fontsize=8)
plt.show()



directory_hemoglobin = directory + r'\Brain data'
name = f'Artinis_S1'
data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_22_events_plot(directory_hemoglobin, name)
print(data)
print(fs)
print(list_indices)
print(list_time_events)
print(pre_event_indices)
print(derived_end_indices)
print(final_event_indices)
print(list_training_sets)


for i in range(len(list_training_sets)):
    training_set = list_training_sets[i]
    # I will check it out but hypothetically the Rx1 and Rx3 are the long distance receiver
    # I will check it out but hypothetically the 9322 is left side and the 9323 is right side
    # print(training_set.columns)
    print(f'Set is {i+1}')

    time = training_set['Time'].to_numpy()
    left_Rx1_TSI_perc = training_set['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'].to_numpy()
    left_RX1_TSI_Fit_Factor = training_set['[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor'].to_numpy()
    left_Rx1_Tx1_O2Hb = training_set['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = training_set['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = training_set['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
    left_Rx2_Tx1_O2Hb = training_set['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
    left_Rx2_Tx2_O2Hb = training_set['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
    left_Rx2_Tx3_O2Hb = training_set['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
    left_Rx1_Tx1_HHb = training_set['[9322] Rx1 - Tx1  HHb'].to_numpy()
    left_Rx1_Tx2_HHb = training_set['[9322] Rx1 - Tx2  HHb'].to_numpy()
    left_Rx1_Tx3_HHb = training_set['[9322] Rx1 - Tx3  HHb'].to_numpy()
    left_Rx2_Tx1_HHb = training_set['[9322] Rx2 - Tx1  HHb'].to_numpy()
    left_Rx2_Tx2_HHb = training_set['[9322] Rx2 - Tx2  HHb'].to_numpy()
    left_Rx2_Tx3_HHb = training_set['[9322] Rx2 - Tx3  HHb'].to_numpy()

    right_Rx3_TSI_perc = training_set['[9323] Rx3 - Tx4,Tx5,Tx6  TSI%'].to_numpy()
    right_RX3_TSI_Fit_Factor = training_set['[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor'].to_numpy()
    right_Rx3_Tx4_O2Hb = training_set['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = training_set['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = training_set['[9323] Rx3 - Tx6  O2Hb'].to_numpy()
    right_Rx4_Tx4_O2Hb = training_set['[9323] Rx4 - Tx4  O2Hb'].to_numpy()
    right_Rx4_Tx5_O2Hb = training_set['[9323] Rx4 - Tx5  O2Hb'].to_numpy()
    right_Rx4_Tx6_O2Hb = training_set['[9323] Rx4 - Tx6  O2Hb'].to_numpy()
    right_Rx3_Tx4_HHb = training_set['[9323] Rx3 - Tx4  HHb'].to_numpy()
    right_Rx3_Tx5_HHb = training_set['[9323] Rx3 - Tx5  HHb'].to_numpy()
    right_Rx3_Tx6_HHb = training_set['[9323] Rx3 - Tx6  HHb'].to_numpy()
    right_Rx4_Tx4_HHb = training_set['[9323] Rx4 - Tx4  HHb'].to_numpy()
    right_Rx4_Tx5_HHb = training_set['[9323] Rx4 - Tx5  HHb'].to_numpy()
    right_Rx4_Tx6_HHb = training_set['[9323] Rx4 - Tx6  HHb'].to_numpy()

    plot_time = time - time[0]
    plt.plot(plot_time, left_RX1_TSI_Fit_Factor, label='left TSI Fit Factor', lw=3)
    plt.plot(plot_time, right_RX3_TSI_Fit_Factor, label='right TSI Fit Factor', lw=3)
    plt.axhline(y=90, label='Threshold of accurate data', c='red', lw=3)
    plt.legend()
    plt.show()

    if i+1 == 8:
        plot = False
    else:
        plot = False
    evaluation_left_Rx1_Tx1_O2Hb, peak_height_left_Rx1_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx1_O2Hb, 100, '[9322] Rx1 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx2_O2Hb, peak_height_left_Rx1_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx2_O2Hb, 100, '[9322] Rx1 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx3_O2Hb, peak_height_left_Rx1_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx3_O2Hb, 100, '[9322] Rx1 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx1_O2Hb, peak_height_left_Rx2_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx1_O2Hb, 100, '[9322] Rx2 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx2_O2Hb, peak_height_left_Rx2_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx2_O2Hb, 100, '[9322] Rx2 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx3_O2Hb, peak_height_left_Rx2_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx3_O2Hb, 100, '[9322] Rx2 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx1_HHb, peak_height_left_Rx1_Tx1_HHb = lb.fNIRS_check_quality(left_Rx1_Tx1_HHb, 100, '[9322] Rx1 - Tx1  HHb', plot=plot)
    evaluation_left_Rx1_Tx2_HHb, peak_height_left_Rx1_Tx2_HHb = lb.fNIRS_check_quality(left_Rx1_Tx2_HHb, 100, '[9322] Rx1 - Tx2  HHb', plot=plot)
    evaluation_left_Rx1_Tx3_HHb, peak_height_left_Rx1_Tx3_HHb = lb.fNIRS_check_quality(left_Rx1_Tx3_HHb, 100, '[9322] Rx1 - Tx3  HHb', plot=plot)
    evaluation_left_Rx2_Tx1_HHb, peak_height_left_Rx2_Tx1_HHb = lb.fNIRS_check_quality(left_Rx2_Tx1_HHb, 100, '[9322] Rx2 - Tx1  HHb', plot=plot)
    evaluation_left_Rx2_Tx2_HHb, peak_height_left_Rx2_Tx2_HHb = lb.fNIRS_check_quality(left_Rx2_Tx2_HHb, 100, '[9322] Rx2 - Tx2  HHb', plot=plot)
    evaluation_left_Rx2_Tx3_HHb, peak_height_left_Rx2_Tx3_HHb = lb.fNIRS_check_quality(left_Rx2_Tx3_HHb, 100, '[9322] Rx2 - Tx3  HHb', plot=plot)

    evaluation_right_Rx3_Tx4_O2Hb, peak_height_right_Rx3_Tx4_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx4_O2Hb, 100, '[9323] Rx3 - Tx4  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx5_O2Hb, peak_height_right_Rx3_Tx5_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx5_O2Hb, 100, '[9323] Rx3 - Tx5  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx6_O2Hb, peak_height_right_Rx3_Tx6_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx6_O2Hb, 100, '[9323] Rx3 - Tx6  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx4_O2Hb, peak_height_right_Rx4_Tx4_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx4_O2Hb, 100, '[9323] Rx4 - Tx4  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx5_O2Hb, peak_height_right_Rx4_Tx5_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx5_O2Hb, 100, '[9323] Rx4 - Tx5  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx6_O2Hb, peak_height_right_Rx4_Tx6_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx6_O2Hb, 100, '[9323] Rx4 - Tx6  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx4_HHb, peak_height_right_Rx3_Tx4_HHb = lb.fNIRS_check_quality(right_Rx3_Tx4_HHb, 100, '[9323] Rx3 - Tx4  HHb', plot=plot)
    evaluation_right_Rx3_Tx5_HHb, peak_height_right_Rx3_Tx5_HHb = lb.fNIRS_check_quality(right_Rx3_Tx5_HHb, 100, '[9323] Rx3 - Tx5  HHb', plot=plot)
    evaluation_right_Rx3_Tx6_HHb, peak_height_right_Rx3_Tx6_HHb = lb.fNIRS_check_quality(right_Rx3_Tx6_HHb, 100, '[9323] Rx3 - Tx6  HHb', plot=plot)
    evaluation_right_Rx4_Tx4_HHb, peak_height_right_Rx4_Tx4_HHb = lb.fNIRS_check_quality(right_Rx4_Tx4_HHb, 100, '[9323] Rx4 - Tx4  HHb', plot=plot)
    evaluation_right_Rx4_Tx5_HHb, peak_height_right_Rx4_Tx5_HHb = lb.fNIRS_check_quality(right_Rx4_Tx5_HHb, 100, '[9323] Rx4 - Tx5  HHb', plot=plot)
    evaluation_right_Rx4_Tx6_HHb, peak_height_right_Rx4_Tx6_HHb = lb.fNIRS_check_quality(right_Rx4_Tx6_HHb, 100, '[9323] Rx4 - Tx6  HHb', plot=plot)

    list_evaluation_left = [evaluation_left_Rx1_Tx1_O2Hb, evaluation_left_Rx1_Tx2_O2Hb, evaluation_left_Rx1_Tx3_O2Hb, evaluation_left_Rx2_Tx1_O2Hb, evaluation_left_Rx2_Tx2_O2Hb, evaluation_left_Rx2_Tx3_O2Hb, evaluation_left_Rx1_Tx1_HHb, evaluation_left_Rx1_Tx2_HHb, evaluation_left_Rx1_Tx3_HHb, evaluation_left_Rx2_Tx1_HHb, evaluation_left_Rx2_Tx2_HHb, evaluation_left_Rx2_Tx3_HHb]
    list_peak_height_left = [peak_height_left_Rx1_Tx1_O2Hb, peak_height_left_Rx1_Tx2_O2Hb, peak_height_left_Rx1_Tx3_O2Hb, peak_height_left_Rx2_Tx1_O2Hb, peak_height_left_Rx2_Tx2_O2Hb, peak_height_left_Rx2_Tx3_O2Hb, peak_height_left_Rx1_Tx1_HHb, peak_height_left_Rx1_Tx2_HHb, peak_height_left_Rx1_Tx3_HHb, peak_height_left_Rx2_Tx1_HHb, peak_height_left_Rx2_Tx2_HHb, peak_height_left_Rx2_Tx3_HHb]
    list_evaluation_right = [evaluation_right_Rx3_Tx4_O2Hb, evaluation_right_Rx3_Tx5_O2Hb, evaluation_right_Rx3_Tx6_O2Hb, evaluation_right_Rx4_Tx4_O2Hb, evaluation_right_Rx4_Tx5_O2Hb, evaluation_right_Rx4_Tx6_O2Hb, evaluation_right_Rx3_Tx4_HHb, evaluation_right_Rx3_Tx5_HHb, evaluation_right_Rx3_Tx6_HHb, evaluation_right_Rx4_Tx4_HHb, evaluation_right_Rx4_Tx5_HHb, evaluation_right_Rx4_Tx6_HHb]
    list_peak_height_right = [peak_height_right_Rx3_Tx4_O2Hb, peak_height_right_Rx3_Tx5_O2Hb, peak_height_right_Rx3_Tx6_O2Hb, peak_height_right_Rx4_Tx4_O2Hb, peak_height_right_Rx4_Tx5_O2Hb, peak_height_right_Rx4_Tx6_O2Hb, peak_height_right_Rx3_Tx4_HHb, peak_height_right_Rx3_Tx5_HHb, peak_height_right_Rx3_Tx6_HHb, peak_height_right_Rx4_Tx4_HHb, peak_height_right_Rx4_Tx5_HHb, peak_height_right_Rx4_Tx6_HHb]

    plt.plot(list_peak_height_left, label='left')
    plt.plot(list_peak_height_right, label='right')
    plt.axhline(y=12, label='Threshold of accurate cardiac rhythm', c='red')
    plt.legend()
    plt.show()



    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx1_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx2_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx3_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx1_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx2_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx3_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx1_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx2_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx3_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx1_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx2_HHb, fs=fs, thresh_z=4, plot=plot)

    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx4_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx5_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx6_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx4_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx5_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx6_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx4_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx5_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx6_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx4_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx5_HHb, fs=fs, thresh_z=4, plot=plot)


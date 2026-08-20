import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


participants_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals'
os.chdir(participants_directory)
participants = pd.read_excel(r'Participants.xlsx')



directory = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen"

fs = 100

brain_data = {}
list_ID = []
list_group_ID = []
for folder in os.listdir(directory):
    ID = str(folder)
    list_ID.append(ID)
    print(ID)
    group = folder.split("_")[0]
    list_group_ID.append(group)

    brain_folder = os.path.join(directory, folder, "Brain data")
    os.chdir(brain_folder)
    name = "Artinis_" + ID[0] + ID.split("_")[1]

    data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_10_events_plot(brain_folder, name, write_manual_events_to_excel=False, plot=False)
    brain_data[ID] = {
        'Group': group,
        'Sampling Frequency': fs,
        'Raw Data': data,
        'Event Indices': list_indices,
        'Event Times': list_time_events,
        'Pre Event Indices': pre_event_indices,
        'Derived End Indices': derived_end_indices,
        'Final Event Indices': final_event_indices,
        'Raw Training Sets': list_training_sets
    }

    time = data["Time"].to_numpy()

    ###########################
    #### Filtering Process ####
    ###########################
    filter_start_index = max(0, int(list_indices[0] - 60 * fs))
    filter_end_index = min(data.height, int(derived_end_indices[-1] + 60 * fs + 1))
    number_of_rows = filter_end_index - filter_start_index

    data_to_filter = data.slice(filter_start_index, number_of_rows)

    left_Rx1_Tx1_O2Hb = data_to_filter['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = data_to_filter['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = data_to_filter['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
    left_Rx2_Tx1_O2Hb = data_to_filter['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
    left_Rx2_Tx2_O2Hb = data_to_filter['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
    left_Rx2_Tx3_O2Hb = data_to_filter['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
    left_Rx1_Tx1_HHb = data_to_filter['[9322] Rx1 - Tx1  HHb'].to_numpy()
    left_Rx1_Tx2_HHb = data_to_filter['[9322] Rx1 - Tx2  HHb'].to_numpy()
    left_Rx1_Tx3_HHb = data_to_filter['[9322] Rx1 - Tx3  HHb'].to_numpy()
    left_Rx2_Tx1_HHb = data_to_filter['[9322] Rx2 - Tx1  HHb'].to_numpy()
    left_Rx2_Tx2_HHb = data_to_filter['[9322] Rx2 - Tx2  HHb'].to_numpy()
    left_Rx2_Tx3_HHb = data_to_filter['[9322] Rx2 - Tx3  HHb'].to_numpy()
    right_Rx3_Tx4_O2Hb = data_to_filter['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = data_to_filter['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = data_to_filter['[9323] Rx3 - Tx6  O2Hb'].to_numpy()
    right_Rx4_Tx4_O2Hb = data_to_filter['[9323] Rx4 - Tx4  O2Hb'].to_numpy()
    right_Rx4_Tx5_O2Hb = data_to_filter['[9323] Rx4 - Tx5  O2Hb'].to_numpy()
    right_Rx4_Tx6_O2Hb = data_to_filter['[9323] Rx4 - Tx6  O2Hb'].to_numpy()
    right_Rx3_Tx4_HHb = data_to_filter['[9323] Rx3 - Tx4  HHb'].to_numpy()
    right_Rx3_Tx5_HHb = data_to_filter['[9323] Rx3 - Tx5  HHb'].to_numpy()
    right_Rx3_Tx6_HHb = data_to_filter['[9323] Rx3 - Tx6  HHb'].to_numpy()
    right_Rx4_Tx4_HHb = data_to_filter['[9323] Rx4 - Tx4  HHb'].to_numpy()
    right_Rx4_Tx5_HHb = data_to_filter['[9323] Rx4 - Tx5  HHb'].to_numpy()
    right_Rx4_Tx6_HHb = data_to_filter['[9323] Rx4 - Tx6  HHb'].to_numpy()

    left_Rx1_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    #
    # ax1.plot(left_Rx2_Tx1_O2Hb, label="left_Rx2_Tx1_O2Hb")
    # ax1.plot(left_Rx2_Tx2_O2Hb, label="left_Rx2_Tx2_O2Hb")
    # ax1.plot(left_Rx2_Tx3_O2Hb, label="left_Rx2_Tx3_O2Hb")
    # ax1.plot(right_Rx4_Tx4_O2Hb, label="right_Rx4_Tx4_O2Hb")
    # ax1.plot(right_Rx4_Tx5_O2Hb, label="right_Rx4_Tx5_O2Hb")
    # ax1.plot(right_Rx4_Tx6_O2Hb, label="right_Rx4_Tx6_O2Hb")
    # ax1.plot(left_Rx1_Tx1_O2Hb, label="left_Rx1_Tx1_O2Hb", color="black", linewidth=2)
    # ax1.set_ylabel("HbO")
    # ax1.set_title("Filtered HbO signals")
    # ax1.legend()
    # ax1.grid(alpha=0.3)
    #
    # ax2.plot(left_Rx2_Tx1_HHb, label="left_Rx2_Tx1_HHb")
    # ax2.plot(left_Rx2_Tx2_HHb, label="left_Rx2_Tx2_HHb")
    # ax2.plot(left_Rx2_Tx3_HHb, label="left_Rx2_Tx3_HHb")
    # ax2.plot(right_Rx4_Tx4_HHb, label="right_Rx4_Tx4_HHb")
    # ax2.plot(right_Rx4_Tx5_HHb, label="right_Rx4_Tx5_HHb")
    # ax2.plot(right_Rx4_Tx6_HHb, label="right_Rx4_Tx6_HHb")
    # ax2.set_xlabel("Sample")
    # ax2.set_ylabel("HHb")
    # ax2.set_title("Filtered HHb signals")
    # ax2.legend()
    # ax2.grid(alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    #####################
    #### PCA Process ####
    #####################
    analysis_start_index = max(0, int(list_indices[0] - 30 * fs))
    analysis_end_index = min(data.height, int(derived_end_indices[-1] + 1))
    analysis_start_index_filtered = analysis_start_index - filter_start_index
    analysis_end_index_filtered = analysis_end_index - filter_start_index
    analysis_slice = slice(analysis_start_index_filtered, analysis_end_index_filtered)

    short_channel_matrix = np.column_stack([
        left_Rx2_Tx1_O2Hb[analysis_slice],  # Left short HbO 1
        left_Rx2_Tx2_O2Hb[analysis_slice],  # Left short HbO 2
        left_Rx2_Tx3_O2Hb[analysis_slice],  # Left short HbO 3
        right_Rx4_Tx4_O2Hb[analysis_slice],  # Right short HbO 1
        right_Rx4_Tx5_O2Hb[analysis_slice],  # Right short HbO 2
        right_Rx4_Tx6_O2Hb[analysis_slice],  # Right short HbO 3
        left_Rx2_Tx1_HHb[analysis_slice],  # Left short HHb 1
        left_Rx2_Tx2_HHb[analysis_slice],  # Left short HHb 2
        left_Rx2_Tx3_HHb[analysis_slice],  # Left short HHb 3
        right_Rx4_Tx4_HHb[analysis_slice],  # Right short HHb 1
        right_Rx4_Tx5_HHb[analysis_slice],  # Right short HHb 2
        right_Rx4_Tx6_HHb[analysis_slice]  # Right short HHb 3
    ])
    scaler = StandardScaler()
    short_channel_matrix_standardized = scaler.fit_transform(short_channel_matrix)
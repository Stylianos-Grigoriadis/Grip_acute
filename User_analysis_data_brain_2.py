import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import os



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
    #### Assess TSI Factor ####
    ###########################
    # left_RX1_TSI_Fit_Factor = data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor'].to_numpy()
    # right_RX3_TSI_Fit_Factor = data['[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor'].to_numpy()
    #
    # plt.figure(figsize=(14, 5))
    #
    # plt.plot(time, left_RX1_TSI_Fit_Factor, label="Left Fit Factor")
    # plt.plot(time, right_RX3_TSI_Fit_Factor, label="Right Fit Factor")
    # plt.axhline(y=90, color="red", linestyle="--", label="Threshold = 90")
    # plt.xlabel("Time (s)")
    # plt.ylabel("TSI Fit Factor")
    # plt.title(f"TSI Fit Factor – {ID}")
    # plt.legend()
    # plt.grid()
    # plt.show()



    #############################################
    #### Assess The Cardiac rhythm existence ####
    #############################################
    signals_name_columns = ['[9322] Rx1 - Tx1  O2Hb', '[9322] Rx1 - Tx2  O2Hb', '[9322] Rx1 - Tx3  O2Hb', '[9322] Rx2 - Tx1  O2Hb', '[9322] Rx2 - Tx2  O2Hb', '[9322] Rx2 - Tx3  O2Hb', '[9322] Rx1 - Tx1  HHb', '[9322] Rx1 - Tx2  HHb', '[9322] Rx1 - Tx3  HHb', '[9322] Rx2 - Tx1  HHb', '[9322] Rx2 - Tx2  HHb', '[9322] Rx2 - Tx3  HHb', '[9323] Rx3 - Tx4  O2Hb', '[9323] Rx3 - Tx5  O2Hb', '[9323] Rx3 - Tx6  O2Hb', '[9323] Rx4 - Tx4  O2Hb', '[9323] Rx4 - Tx5  O2Hb', '[9323] Rx4 - Tx6  O2Hb', '[9323] Rx3 - Tx4  HHb', '[9323] Rx3 - Tx5  HHb', '[9323] Rx3 - Tx6  HHb', '[9323] Rx4 - Tx4  HHb', '[9323] Rx4 - Tx5  HHb', '[9323] Rx4 - Tx6  HHb']

    evaluation_test_list = []
    peak_height_list = []
    for signal_name in signals_name_columns:
        plot = False

        if ID == "Pink_4" and signal_name== '[9323] Rx3 - Tx4  HHb':
            plot = True

        evaluation_test, peak_height = lb.fNIRS_check_quality(data[signal_name].to_numpy(), 100, signal_name, plot=plot)
        evaluation_test_list.append(evaluation_test)
        peak_height_list.append(peak_height)

    if ID == "Pink_4":
        plt.plot(peak_height_list, marker="o")

        plt.xticks(
            range(len(signals_name_columns)),
            signals_name_columns,
            rotation=90
        )
        plt.axhline(y=12, color='red')

        plt.ylabel("Peak Height of Cardiac assessment")
        plt.tight_layout()

        plt.plot(peak_height_list)
        plt.show()


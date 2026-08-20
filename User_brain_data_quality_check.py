import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import os



participants_directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals'
os.chdir(participants_directory)
participants = pd.read_excel(r'Participants.xlsx')



directory = r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen"

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
    # signals_name_columns = ['[9322] Rx1 - Tx1  O2Hb', '[9322] Rx1 - Tx2  O2Hb', '[9322] Rx1 - Tx3  O2Hb', '[9322] Rx2 - Tx1  O2Hb', '[9322] Rx2 - Tx2  O2Hb', '[9322] Rx2 - Tx3  O2Hb', '[9323] Rx3 - Tx4  O2Hb', '[9323] Rx3 - Tx5  O2Hb', '[9323] Rx3 - Tx6  O2Hb', '[9323] Rx4 - Tx4  O2Hb', '[9323] Rx4 - Tx5  O2Hb', '[9323] Rx4 - Tx6  O2Hb']
    #
    # evaluation_test_list = []
    # peak_height_list = []
    # max_min_peak_height_list = []
    # for signal_name in signals_name_columns:
    #     plot = False
    #
    #     if ID == "White_5" :
    #         plot = True
    #
    #     evaluation_test, peak_height, max_min_peak_height = lb.fNIRS_check_quality(data[signal_name].to_numpy(), 100, signal_name, plot=plot)
    #     evaluation_test_list.append(evaluation_test)
    #     peak_height_list.append(peak_height)
    #     max_min_peak_height_list.append(max_min_peak_height)
    # if ID == "White_5":
    #     plt.plot(peak_height_list, marker="o", label="Gaussian peak")
    #     plt.plot(max_min_peak_height_list, marker="o", label="Actual max-min")
    #     plt.xticks(
    #         range(len(signals_name_columns)),
    #         signals_name_columns,
    #         rotation=90)
    #     plt.axhline(y=12, color='red')
    #     plt.ylabel("Peak Height of Cardiac assessment")
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()

    #################################
    #### Assess Motion artifacts ####
    #################################
    # signals_name_columns = ['[9322] Rx1 - Tx1  O2Hb', '[9322] Rx1 - Tx2  O2Hb', '[9322] Rx1 - Tx3  O2Hb', '[9322] Rx2 - Tx1  O2Hb', '[9322] Rx2 - Tx2  O2Hb', '[9322] Rx2 - Tx3  O2Hb', '[9323] Rx3 - Tx4  O2Hb', '[9323] Rx3 - Tx5  O2Hb', '[9323] Rx3 - Tx6  O2Hb', '[9323] Rx4 - Tx4  O2Hb', '[9323] Rx4 - Tx5  O2Hb', '[9323] Rx4 - Tx6  O2Hb']
    # motion_masks = []
    # motion_z_scores = []
    #
    # for signal_name in signals_name_columns:
    #     mask, z = lb.detect_motion_mask_from_movstd(time_window=2, signal=data[signal_name].to_numpy(), fs=fs,
    #                                                 thresh_z=4, plot=False)
    #
    #     motion_masks.append(mask)
    #     motion_z_scores.append(z)
    #
    # motion_masks = np.array(motion_masks)
    # motion_z_scores = np.array(motion_z_scores)
    #
    # number_of_samples = motion_masks.shape[1]
    #
    # time_tick_positions = np.linspace(0, number_of_samples - 1, 10, dtype=int)
    # time_tick_labels = np.round(time_tick_positions / fs, 1)
    #
    # fig, ax = plt.subplots(figsize=(15, 7))
    #
    # # Create a transparent image:
    # # no motion = transparent
    # # detected motion = red
    # motion_colors = np.zeros((motion_masks.shape[0], motion_masks.shape[1], 4))
    #
    # motion_colors[motion_masks] = [0.8, 0.0, 0.0, 1.0]
    #
    # # Shade the baseline and training periods
    # for set_number, (pre_index, training_index, end_index) in enumerate(
    #         zip(pre_event_indices, list_indices, derived_end_indices), start=1):
    #     # Ten-second baseline
    #     ax.axvspan(pre_index, training_index, color="lightskyblue", alpha=0.35, zorder=0)
    #
    #     # Thirty-second training set
    #     ax.axvspan(training_index, end_index, color="lightgreen", alpha=0.35, zorder=0)
    #
    #     # Training-set number
    #     ax.text((training_index + end_index) / 2, -0.7, f"Set {set_number}", ha="center", va="bottom", fontsize=8,
    #             clip_on=False)
    #
    # # Plot detected motion above the shaded periods
    # ax.imshow(motion_colors, aspect="auto", interpolation="nearest", zorder=2)
    #
    # ax.set_yticks(range(len(signals_name_columns)))
    # ax.set_yticklabels(signals_name_columns)
    #
    # ax.set_xticks(time_tick_positions)
    # ax.set_xticklabels(time_tick_labels)
    #
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Signal")
    # ax.set_title(f"Detected Motion Artifacts – {ID}", pad=35)
    #
    # legend_elements = [
    #     plt.Rectangle((0, 0), 1, 1, color="lightskyblue", alpha=0.35, label="Baseline (10 s)"),
    #     plt.Rectangle((0, 0), 1, 1, color="lightgreen", alpha=0.35, label="Training set (30 s)"),
    #     plt.Rectangle((0, 0), 1, 1, color="red", label="Detected motion")
    # ]
    #
    # ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    #
    # plt.tight_layout()
    # plt.show()
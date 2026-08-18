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
low_pass_filter_frequency = 15
m = 2
r = 0.2

results = []
list_ID = []
list_group_ID = []
for folder in os.listdir(directory):
    ID = str(folder)
    list_ID.append(ID)
    print(ID)
    group = folder.split("_")[0]
    list_group_ID.append(group)

    MVC = participants.loc[participants["ID"] == ID, participants.columns[6]].iloc[0]
    print(MVC)
    grip_folder = os.path.join(directory, folder, "Grip data")
    os.chdir(grip_folder)

    Training_1 = pd.read_csv("Training_1.csv", skiprows=2)
    Training_2 = pd.read_csv("Training_2.csv", skiprows=2)
    Training_3 = pd.read_csv("Training_3.csv", skiprows=2)
    Training_4 = pd.read_csv("Training_4.csv", skiprows=2)
    Training_5 = pd.read_csv("Training_5.csv", skiprows=2)
    Training_6 = pd.read_csv("Training_6.csv", skiprows=2)
    Training_7 = pd.read_csv("Training_7.csv", skiprows=2)
    Training_8 = pd.read_csv("Training_8.csv", skiprows=2)
    Training_9 = pd.read_csv("Training_9.csv", skiprows=2)
    Training_10 = pd.read_csv("Training_10.csv", skiprows=2)

    training_signals = [Training_1, Training_2, Training_3, Training_4, Training_5, Training_6, Training_7, Training_8, Training_9, Training_10]

    #############################
    ##### Residual Analysis #####
    #############################
    # cutoff_frequencies = np.arange(1, 50)
    # signal_names = ["Training_1", "Training_2", "Training_3", "Training_4", "Training_5", "Training_6", "Training_7", "Training_8", "Training_9", "Training_10"]
    # list_residual_FC = []
    # for signal in training_signals:
    #     signal_Fc = lib.Residual_analysis(signal["Performance"], cutoff_frequencies, fs, number_of_fit_points=10, plot=False, save=None)
    #     list_residual_FC.append(signal_Fc)
    # plt.plot(list_residual_FC, marker="o")
    #
    # plt.xticks(
    #     range(len(signal_names)),
    #     signal_names,
    #     rotation=90
    # )
    #
    # plt.ylabel("Optimal Cutoff frequency (Hz)")
    # plt.tight_layout()
    # # save_name = ID + ".png"
    # # plt.savefig(
    # #     os.path.join(r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Figures\Residual plots Grip data', save_name),
    # #     dpi=300,
    # #     bbox_inches="tight"
    # # )
    # plt.close()


    #####################
    ##### Filtering #####
    #####################
    for signal in training_signals:
        signal['Performance'] = lib.Butterworth(fs, low_pass_filter_frequency, signal['Performance'])

    ################################
    ##### Syncing with targets #####
    ################################
    Sync_Training_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_1)
    Sync_Training_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_2)
    Sync_Training_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_3)
    Sync_Training_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_4)
    Sync_Training_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_5)
    Sync_Training_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_6)
    Sync_Training_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_7)
    Sync_Training_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_8)
    Sync_Training_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_9)
    Sync_Training_10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Training_10)

    #################################
    ##### Exclude first seconds #####
    #################################
    time_threshold = 2
    Sync_Training_1 = Sync_Training_1[Sync_Training_1['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_2 = Sync_Training_2[Sync_Training_2['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_3 = Sync_Training_3[Sync_Training_3['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_4 = Sync_Training_4[Sync_Training_4['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_5 = Sync_Training_5[Sync_Training_5['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_6 = Sync_Training_6[Sync_Training_6['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_7 = Sync_Training_7[Sync_Training_7['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_8 = Sync_Training_8[Sync_Training_8['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_9 = Sync_Training_9[Sync_Training_9['Time'] > time_threshold].reset_index(drop=True).copy()
    Sync_Training_10 = Sync_Training_10[Sync_Training_10['Time'] > time_threshold].reset_index(drop=True).copy()

    #######################################
    ##### Training Sets data analysis #####
    #######################################
    Sync_training_signals = [Sync_Training_1, Sync_Training_2, Sync_Training_3, Sync_Training_4, Sync_Training_5, Sync_Training_6, Sync_Training_7, Sync_Training_8, Sync_Training_9, Sync_Training_10]

    participant_results = {
        "ID": ID,
        "Group": group
    }

    for trial, (Sync_training_signal, training_signal) in enumerate(zip(Sync_training_signals, training_signals), start=1):
        spatial_error = lb.spatial_error(Sync_training_signal)
        average_spatial_error = np.mean(spatial_error)
        variable_error = np.std(spatial_error)
        relative_to_MVC_average_spatial_error = (average_spatial_error/MVC)*100
        relative_to_MVC_variable_error = (variable_error/MVC)*100

        AMI = lib.AMI_Stergiou(training_signal['Performance'].to_numpy(), 10, fs, n_bins=0, plot=False)
        SaEn = lib.SaEn_once_again(training_signal['Performance'].to_numpy(), m, r, AMI[0][0][0], return_probabilities=False)

        lag, max_correlation = lib.Cross_correlation(Sync_training_signal["Target"], Sync_training_signal["Performance"], plot=False)

        participant_results[f"Average Spatial Error Training {trial}"] = average_spatial_error
        participant_results[f"Variable Error Training {trial}"] = variable_error
        participant_results[f"Normalized Average Spatial Error Training {trial}"] = relative_to_MVC_average_spatial_error
        participant_results[f"Normalized Variable Error Training {trial}"] = relative_to_MVC_variable_error
        participant_results[f"SaEn Training {trial}"] = SaEn
        participant_results[f"Sample Lag Training {trial}"] = lag
        participant_results[f"Max Correlation Training {trial}"] = max_correlation

    results.append(participant_results)

df_results = pd.DataFrame(results)
save_directory = r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"
df_results.to_excel(os.path.join(save_directory, "Training_results.xlsx"),
    index=False
)
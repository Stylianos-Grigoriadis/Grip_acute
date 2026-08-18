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
sd_factor = 3
time_window = 1
asymptote_fraction = 0.95

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


    Isometric_high = pd.read_csv("Isometric_high.csv", skiprows=2)
    Isometric_low = pd.read_csv("Isometric_low.csv", skiprows=2)
    Pre_Pert_down_1 = pd.read_csv("Pre_Pert_down_1.csv", skiprows=2)
    Pre_Pert_down_2 = pd.read_csv("Pre_Pert_down_2.csv", skiprows=2)
    Pre_Pert_down_3 = pd.read_csv("Pre_Pert_down_3.csv", skiprows=2)
    Pre_Pert_up_1 = pd.read_csv("Pre_Pert_up_1.csv", skiprows=2)
    Pre_Pert_up_2 = pd.read_csv("Pre_Pert_up_2.csv", skiprows=2)
    Pre_Pert_up_3 = pd.read_csv("Pre_Pert_up_3.csv", skiprows=2)
    Post_Pert_down_1 = pd.read_csv("Post_Pert_down_1.csv", skiprows=2)
    Post_Pert_down_2 = pd.read_csv("Post_Pert_down_2.csv", skiprows=2)
    Post_Pert_down_3 = pd.read_csv("Post_Pert_down_3.csv", skiprows=2)
    Post_Pert_up_1 = pd.read_csv("Post_Pert_up_1.csv", skiprows=2)
    Post_Pert_up_2 = pd.read_csv("Post_Pert_up_2.csv", skiprows=2)
    Post_Pert_up_3 = pd.read_csv("Post_Pert_up_3.csv", skiprows=2)

    all_signals = [Isometric_high, Isometric_low, Pre_Pert_down_1, Pre_Pert_down_2, Pre_Pert_down_3, Pre_Pert_up_1, Pre_Pert_up_2, Pre_Pert_up_3, Post_Pert_down_1, Post_Pert_down_2, Post_Pert_down_3, Post_Pert_up_1, Post_Pert_up_2, Post_Pert_up_3]

    #####################
    ##### Filtering #####
    #####################
    for signal in all_signals:
        signal['Performance'] = lib.Butterworth(fs, low_pass_filter_frequency, signal['Performance'])

    ##########################################################
    ##### Calculate average and sd from isometric trials #####
    ##########################################################
    synch_Isometric_high = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Isometric_high)
    synch_Isometric_low = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(Isometric_low)

    time_threshold = 3
    synch_Isometric_high_after_threshold = synch_Isometric_high[synch_Isometric_high['Time'] > time_threshold].reset_index(drop=True).copy()
    synch_Isometric_low_after_threshold = synch_Isometric_low[synch_Isometric_low['Time'] > time_threshold].reset_index(drop=True).copy()
    spatial_errors_synch_Isometric_high = lb.spatial_error(synch_Isometric_high_after_threshold)
    spatial_errors_synch_Isometric_low = lb.spatial_error(synch_Isometric_low_after_threshold)
    mean_spatial_error_synch_Isometric_high = np.mean(spatial_errors_synch_Isometric_high)
    mean_spatial_error_synch_Isometric_low = np.mean(spatial_errors_synch_Isometric_low)
    sd_spatial_error_synch_Isometric_high = np.std(spatial_errors_synch_Isometric_high)
    sd_spatial_error_synch_Isometric_low = np.std(spatial_errors_synch_Isometric_low)


    #########################################
    ##### Calculate the adaptation time #####
    #########################################

    Pre_Pert_down_1_time_to_adapt_sd, Pre_Pert_down_1_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_down_1, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Pre_Pert_down_2_time_to_adapt_sd, Pre_Pert_down_2_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_down_2, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Pre_Pert_down_3_time_to_adapt_sd, Pre_Pert_down_3_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_down_3, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Pre_Pert_up_1_time_to_adapt_sd, Pre_Pert_up_1_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_up_1, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)
    Pre_Pert_up_2_time_to_adapt_sd, Pre_Pert_up_2_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_up_2, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)
    Pre_Pert_up_3_time_to_adapt_sd, Pre_Pert_up_3_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Pre_Pert_up_3, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_down_1_time_to_adapt_sd, Post_Pert_down_1_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_down_1, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_down_2_time_to_adapt_sd, Post_Pert_down_2_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_down_2, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_down_3_time_to_adapt_sd, Post_Pert_down_3_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_down_3, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_low, sd_spatial_error_synch_Isometric_low, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_up_1_time_to_adapt_sd, Post_Pert_up_1_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_up_1, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_up_2_time_to_adapt_sd, Post_Pert_up_2_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_up_2, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)
    Post_Pert_up_3_time_to_adapt_sd, Post_Pert_up_3_time_to_adapt_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Post_Pert_up_3, sd_factor, time_window, ID, mean_spatial_error_synch_Isometric_high, sd_spatial_error_synch_Isometric_high, asymptote_fraction=asymptote_fraction, plot=False)


    # Functions to safely calculate mean and minimum when some trials return None
    def safe_mean(values):
        values = np.array([np.nan if value is None else value for value in values], dtype=float)

        if np.all(np.isnan(values)):
            return np.nan

        return np.nanmean(values)


    def safe_min(values):
        values = np.array([np.nan if value is None else value for value in values], dtype=float)

        if np.all(np.isnan(values)):
            return np.nan

        return np.nanmin(values)


    ############################################
    ##### Average and minimum - SD method #####
    ############################################
    Average_Pre_Pert_down_time_to_adapt_sd = safe_mean([Pre_Pert_down_1_time_to_adapt_sd, Pre_Pert_down_2_time_to_adapt_sd, Pre_Pert_down_3_time_to_adapt_sd])
    Average_Pre_Pert_up_time_to_adapt_sd = safe_mean([Pre_Pert_up_1_time_to_adapt_sd, Pre_Pert_up_2_time_to_adapt_sd, Pre_Pert_up_3_time_to_adapt_sd])
    Average_Post_Pert_down_time_to_adapt_sd = safe_mean([Post_Pert_down_1_time_to_adapt_sd, Post_Pert_down_2_time_to_adapt_sd, Post_Pert_down_3_time_to_adapt_sd])
    Average_Post_Pert_up_time_to_adapt_sd = safe_mean([Post_Pert_up_1_time_to_adapt_sd, Post_Pert_up_2_time_to_adapt_sd, Post_Pert_up_3_time_to_adapt_sd])
    Min_Pre_Pert_down_time_to_adapt_sd = safe_min([Pre_Pert_down_1_time_to_adapt_sd, Pre_Pert_down_2_time_to_adapt_sd, Pre_Pert_down_3_time_to_adapt_sd])
    Min_Pre_Pert_up_time_to_adapt_sd = safe_min([Pre_Pert_up_1_time_to_adapt_sd, Pre_Pert_up_2_time_to_adapt_sd, Pre_Pert_up_3_time_to_adapt_sd])
    Min_Post_Pert_down_time_to_adapt_sd = safe_min([Post_Pert_down_1_time_to_adapt_sd, Post_Pert_down_2_time_to_adapt_sd, Post_Pert_down_3_time_to_adapt_sd])
    Min_Post_Pert_up_time_to_adapt_sd = safe_min([Post_Pert_up_1_time_to_adapt_sd, Post_Pert_up_2_time_to_adapt_sd, Post_Pert_up_3_time_to_adapt_sd])

    ###################################################
    ##### Average and minimum - Asymptote method #####
    ###################################################

    Average_Pre_Pert_down_time_to_adapt_asymp = safe_mean([Pre_Pert_down_1_time_to_adapt_asymp, Pre_Pert_down_2_time_to_adapt_asymp, Pre_Pert_down_3_time_to_adapt_asymp])
    Average_Pre_Pert_up_time_to_adapt_asymp = safe_mean([Pre_Pert_up_1_time_to_adapt_asymp, Pre_Pert_up_2_time_to_adapt_asymp, Pre_Pert_up_3_time_to_adapt_asymp])
    Average_Post_Pert_down_time_to_adapt_asymp = safe_mean([Post_Pert_down_1_time_to_adapt_asymp, Post_Pert_down_2_time_to_adapt_asymp, Post_Pert_down_3_time_to_adapt_asymp])
    Average_Post_Pert_up_time_to_adapt_asymp = safe_mean([Post_Pert_up_1_time_to_adapt_asymp, Post_Pert_up_2_time_to_adapt_asymp, Post_Pert_up_3_time_to_adapt_asymp])
    Min_Pre_Pert_down_time_to_adapt_asymp = safe_min([Pre_Pert_down_1_time_to_adapt_asymp, Pre_Pert_down_2_time_to_adapt_asymp, Pre_Pert_down_3_time_to_adapt_asymp])
    Min_Pre_Pert_up_time_to_adapt_asymp = safe_min([Pre_Pert_up_1_time_to_adapt_asymp, Pre_Pert_up_2_time_to_adapt_asymp, Pre_Pert_up_3_time_to_adapt_asymp])
    Min_Post_Pert_down_time_to_adapt_asymp = safe_min([Post_Pert_down_1_time_to_adapt_asymp, Post_Pert_down_2_time_to_adapt_asymp, Post_Pert_down_3_time_to_adapt_asymp])
    Min_Post_Pert_up_time_to_adapt_asymp = safe_min([Post_Pert_up_1_time_to_adapt_asymp, Post_Pert_up_2_time_to_adapt_asymp, Post_Pert_up_3_time_to_adapt_asymp])

    ############################
    ##### Save participant #####
    ############################

    participant_results = {
        "ID": ID,
        "Group": group
    }

    adaptation_results = {
        "Pre Down 1": (Pre_Pert_down_1_time_to_adapt_sd, Pre_Pert_down_1_time_to_adapt_asymp),
        "Pre Down 2": (Pre_Pert_down_2_time_to_adapt_sd, Pre_Pert_down_2_time_to_adapt_asymp),
        "Pre Down 3": (Pre_Pert_down_3_time_to_adapt_sd, Pre_Pert_down_3_time_to_adapt_asymp),

        "Pre Up 1": (Pre_Pert_up_1_time_to_adapt_sd, Pre_Pert_up_1_time_to_adapt_asymp),
        "Pre Up 2": (Pre_Pert_up_2_time_to_adapt_sd, Pre_Pert_up_2_time_to_adapt_asymp),
        "Pre Up 3": (Pre_Pert_up_3_time_to_adapt_sd, Pre_Pert_up_3_time_to_adapt_asymp),

        "Post Down 1": (Post_Pert_down_1_time_to_adapt_sd, Post_Pert_down_1_time_to_adapt_asymp),
        "Post Down 2": (Post_Pert_down_2_time_to_adapt_sd, Post_Pert_down_2_time_to_adapt_asymp),
        "Post Down 3": (Post_Pert_down_3_time_to_adapt_sd, Post_Pert_down_3_time_to_adapt_asymp),

        "Post Up 1": (Post_Pert_up_1_time_to_adapt_sd, Post_Pert_up_1_time_to_adapt_asymp),
        "Post Up 2": (Post_Pert_up_2_time_to_adapt_sd, Post_Pert_up_2_time_to_adapt_asymp),
        "Post Up 3": (Post_Pert_up_3_time_to_adapt_sd, Post_Pert_up_3_time_to_adapt_asymp)
    }

    # Save individual perturbation trials
    for condition, (adaptation_sd, adaptation_asymp) in adaptation_results.items():
        if adaptation_sd is None:
            adaptation_sd = np.nan
        if adaptation_asymp is None:
            adaptation_asymp = np.nan
        participant_results[f"{condition} Adaptation Time SD {sd_factor}"] = adaptation_sd
        participant_results[f"{condition} Adaptation Time Asymptote"] = adaptation_asymp

    # Average - SD method
    participant_results[f"Average Pre Down Adaptation Time SD {sd_factor}"] = Average_Pre_Pert_down_time_to_adapt_sd
    participant_results[f"Average Pre Up Adaptation Time SD {sd_factor}"] = Average_Pre_Pert_up_time_to_adapt_sd
    participant_results[f"Average Post Down Adaptation Time SD {sd_factor}"] = Average_Post_Pert_down_time_to_adapt_sd
    participant_results[f"Average Post Up Adaptation Time SD {sd_factor}"] = Average_Post_Pert_up_time_to_adapt_sd

    # Minimum - SD method
    participant_results[f"Minimum Pre Down Adaptation Time SD {sd_factor}"] = Min_Pre_Pert_down_time_to_adapt_sd
    participant_results[f"Minimum Pre Up Adaptation Time SD {sd_factor}"] = Min_Pre_Pert_up_time_to_adapt_sd
    participant_results[f"Minimum Post Down Adaptation Time SD {sd_factor}"] = Min_Post_Pert_down_time_to_adapt_sd
    participant_results[f"Minimum Post Up Adaptation Time SD {sd_factor}"] = Min_Post_Pert_up_time_to_adapt_sd

    # Average - Asymptote method
    participant_results["Average Pre Down Adaptation Time Asymptote"] = Average_Pre_Pert_down_time_to_adapt_asymp
    participant_results["Average Pre Up Adaptation Time Asymptote"] = Average_Pre_Pert_up_time_to_adapt_asymp
    participant_results["Average Post Down Adaptation Time Asymptote"] = Average_Post_Pert_down_time_to_adapt_asymp
    participant_results["Average Post Up Adaptation Time Asymptote"] = Average_Post_Pert_up_time_to_adapt_asymp

    # Minimum - Asymptote method
    participant_results["Minimum Pre Down Adaptation Time Asymptote"] = Min_Pre_Pert_down_time_to_adapt_asymp
    participant_results["Minimum Pre Up Adaptation Time Asymptote"] = Min_Pre_Pert_up_time_to_adapt_asymp
    participant_results["Minimum Post Down Adaptation Time Asymptote"] = Min_Post_Pert_down_time_to_adapt_asymp
    participant_results["Minimum Post Up Adaptation Time Asymptote"] = Min_Post_Pert_up_time_to_adapt_asymp

    results.append(participant_results)

    ################################
    ##### Save final Excel file #####
    ################################

df_results = pd.DataFrame(results)
save_directory = r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"
save_name = f"Perturbation_results_SD_factor_{sd_factor}_Asymptote_{asymptote_fraction}.xlsx"
df_results.to_excel(os.path.join(save_directory, save_name),index=False)

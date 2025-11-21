import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Force data'
os.chdir(directory)

parts = directory.split(os.sep)
ID = parts[-2]
print(ID)

training_set_1 = pd.read_csv(r'Training_set_1.csv', skiprows=2)
training_set_2 = pd.read_csv(r'Training_set_2.csv', skiprows=2)
training_set_3 = pd.read_csv(r'Training_set_3.csv', skiprows=2)
training_set_4 = pd.read_csv(r'Training_set_4.csv', skiprows=2)
training_set_5 = pd.read_csv(r'Training_set_5.csv', skiprows=2)
training_set_6 = pd.read_csv(r'Training_set_6.csv', skiprows=2)
training_set_7 = pd.read_csv(r'Training_set_7.csv', skiprows=2)
training_set_8 = pd.read_csv(r'Training_set_8.csv', skiprows=2)
training_set_9 = pd.read_csv(r'Training_set_9.csv', skiprows=2)
training_set_with_pert = pd.read_csv(r'Training_set_with_pert.csv', skiprows=2)
list_sets = [training_set_1, training_set_2, training_set_3, training_set_4, training_set_5, training_set_6, training_set_7, training_set_8, training_set_9, training_set_with_pert]

# Residual analysis
list_of_frequencies = range(2,30)
list_RMS = []
for set in list_sets:
    list_RMS_set = []
    for low_pass_filter_frequency in list_of_frequencies:
        filtered_set = lib.Butterworth(75, low_pass_filter_frequency, set['Performance'])
        RMS = lb.RMS(set['Performance'], filtered_set)
        list_RMS_set.append(RMS)
    list_RMS.append(list_RMS_set)

for i in range(len(list_sets)):
    plt.plot(list_of_frequencies, list_RMS[i], label=f'Set {i+1}')
plt.legend()
plt.title(f'Residual analysis for {ID}')
# Save residual plot
save_image_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Figures\Residual plots'
save_path = fr"{save_image_directory}\Residual_analysis_{ID}.png"  # or .jpg, .pdf
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


# Training sets analysis
spatial_error_mean = []
spatial_error_sd = []
for i, set in enumerate(list_sets):

    # Filtering at 10Hz
    set['Performance'] = lib.Butterworth(75,10,set['Performance'])

    # Synchronizing target with performance
    set = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(set)

    # Start calculate spatial error
    # Begin by taking the index of the 2 seconds so that you can protect spatial error from initial extremes
    first_seconds = 2
    idx = (set['Time'] - first_seconds).abs().idxmin()  # index of closest value
    closest_value = set['Time'].iloc[idx]
    print("Index:", idx)
    print("Closest value:", closest_value)

    # Calculate spatial error
    spatial_error = lb.spatial_error(set)

    # Plot the performance vs target and the spatial error
    # fig, ax1 = plt.subplots()
    #
    # # --- Primary axis (left) ---
    # ax1.plot(set['Performance'][:329], label='Force')
    # ax1.plot(set['Target'][:329], label='Target')
    # ax1.set_ylabel("Force / Target")
    # ax1.axvline(x=idx, color='red', label='Start of spatial error', lw=3)
    #
    # # Legend for primary axis
    # ax1.legend(loc="upper left")
    #
    # # --- Secondary axis (right) ---
    # ax2 = ax1.twinx()
    # ax2.plot(spatial_error, color='black', label='Spatial Error')
    # ax2.set_ylabel("Spatial Error")
    #
    # # Legend for secondary axis
    # ax2.legend(loc="upper right")
    #
    # plt.title("Force, Target and Spatial Error")
    # plt.show()

    # Keep the spatial error of each
    if i == 0:
        # Take the last index of the first training set, so that in the perturbation set you only keep the training data
        last_index_of_training_set = len(spatial_error)
        print(last_index_of_training_set)
    spatial_error_mean.append(np.mean(spatial_error[idx:last_index_of_training_set]))
    spatial_error_sd.append(np.std(spatial_error[idx:last_index_of_training_set]))


spatial_error_mean = np.array(spatial_error_mean)
spatial_error_sd = np.array(spatial_error_sd)
# Plot the average spatial error and sd on all sets
# x = np.linspace(1, 10, 10)
# labels = [f"Set {i+1}" for i in range(10)]
# plt.plot(x, spatial_error_mean, label="Mean")
# plt.fill_between(x, spatial_error_mean - spatial_error_sd, spatial_error_mean + spatial_error_sd, alpha=0.3, label="STD")
# plt.legend()
# plt.xticks(x, labels, rotation=45)
# plt.ylabel('Spatial Error')
# plt.show()

# Perturbation analysis
isometric = pd.read_csv(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Isometric\Isometric_80_T2.csv', skiprows=2)
isometric['Performance'] = lib.Butterworth(75, 10, isometric['Performance'])
isometric = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(isometric)
spatial_error_isometric = lb.spatial_error(isometric)
idx = (isometric['Time'] - first_seconds).abs().idxmin()  # index of closest value
closest_value = isometric['Time'].iloc[idx]
mean_spatial_error_isometric = np.mean(spatial_error_isometric[idx:])
sd_spatial_error_isometric = np.std(spatial_error_isometric[idx:])
training_set_with_pert = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_with_pert)
training_set_only_perturbation = training_set_with_pert.iloc[329:].reset_index(drop=True).copy()
sd_factor = 2
consecutive_values = 20

time_to_adapt = lb.adaptation_time_using_sd(training_set_only_perturbation, sd_factor, consecutive_values, ID, mean_spatial_error_isometric, sd_spatial_error_isometric, plot=True)



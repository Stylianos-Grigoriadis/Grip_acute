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
# save_image_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Figures\Residual plots'
# save_path = fr"{save_image_directory}\Residual_analysis_{ID}.png"  # or .jpg, .pdf
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

spatial_error_average = []
spatial_error_sd = []
for set in list_sets:

    # Filtering at 10Hz
    set['Performance'] = lib.Butterworth(75,10,set['Performance'])

    # Synchronizing target with performance
    set = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(set)
    print(len(set['Performance']))
    print(len(set['Target']))

    spatial_error = lb.spatial_error(set)
    plt.plot(set['Performance'][:329], label='Force')
    plt.plot(set['Target'][:329], label='Target')
    plt.legend()
    plt.show()







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

# training_set_1 = pd.read_csv(r'Training_set_1.csv', skiprows=2)
# training_set_2 = pd.read_csv(r'Training_set_2.csv', skiprows=2)
# training_set_3 = pd.read_csv(r'Training_set_3.csv', skiprows=2)
# training_set_4 = pd.read_csv(r'Training_set_4.csv', skiprows=2)
# training_set_5 = pd.read_csv(r'Training_set_5.csv', skiprows=2)
# training_set_6 = pd.read_csv(r'Training_set_6.csv', skiprows=2)
# training_set_7 = pd.read_csv(r'Training_set_7.csv', skiprows=2)
# training_set_8 = pd.read_csv(r'Training_set_8.csv', skiprows=2)
# training_set_9 = pd.read_csv(r'Training_set_9.csv', skiprows=2)
# training_set_with_pert = pd.read_csv(r'Training_set_with_pert.csv', skiprows=2)
#
# training_set_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_1)
# training_set_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_2)
# training_set_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_3)
# training_set_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_4)
# training_set_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_5)
# training_set_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_6)
# training_set_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_7)
# training_set_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_8)
# training_set_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_9)
# training_set_with_pert = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_with_pert)
#
# list_set = [training_set_1,
#             training_set_2,
#             training_set_3,
#             training_set_4,
#             training_set_5,
#             training_set_6,
#             training_set_7,
#             training_set_8,
#             training_set_9,
#             training_set_with_pert
#             ]
# list_average_difference_time_to_ClosestSampleTime = []
# list_std_difference_time_to_ClosestSampleTime = []
# for set in list_set:
#     force_time = set['Time'].to_numpy()
#     target_time = set['ClosestSampleTime'].to_numpy()
#     difference = force_time-target_time
#     set['Difference'] = difference
#     print(set)
#     average = np.average(np.abs(difference))
#     std = np.std(np.abs(difference))
#     print(f'Average is {average}')
#     print(f'Std is {std}')
#     list_average_difference_time_to_ClosestSampleTime.append(average)
#     list_std_difference_time_to_ClosestSampleTime.append(std)
# list_average_difference_time_to_ClosestSampleTime = np.array(list_average_difference_time_to_ClosestSampleTime)
# list_std_difference_time_to_ClosestSampleTime = np.array(list_std_difference_time_to_ClosestSampleTime)
# upper = list_average_difference_time_to_ClosestSampleTime + list_std_difference_time_to_ClosestSampleTime
# lower = list_average_difference_time_to_ClosestSampleTime - list_std_difference_time_to_ClosestSampleTime
#
# x = np.arange(len(list_average_difference_time_to_ClosestSampleTime))
# plt.plot(list_average_difference_time_to_ClosestSampleTime, label='Average Difference', linewidth=2)
# plt.fill_between(x, lower, upper, alpha=0.3, label='±SD')
# labels = [f"Set {i+1}" for i in range(len(list_average_difference_time_to_ClosestSampleTime))]
# plt.xlabel('Time')
# plt.ylabel('Difference')
# plt.title('Difference to ClosestSampleTime with ±SD')
# plt.legend(loc='upper left')
# plt.xticks(x, labels, rotation=45)  # Set custom labels
#
# plt.show()
#
#
# fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
# axes = axes.flatten()  # Make indexing easier
#
# for i, ax in enumerate(axes):
#     set = list_set[i]
#
#     ax.plot(set['Time'], set['Performance'], label='Force Output')
#     ax.plot(set['ClosestSampleTime'], set['Target'], label='Target')
#
#     ax.set_title(f"Set {i + 1}")
#     ax.legend(fontsize=8)
#
# plt.show()

directory_hemoglobin = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Brain data'
name = f'Data_{ID}'
print('HEllo')
brain_data, fs = lb.artinis_read_file_10_sets(directory_hemoglobin, name)




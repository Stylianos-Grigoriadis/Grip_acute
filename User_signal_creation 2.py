import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
import random

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'serif'

# Parameters of the % MVC
desired_sd_MVC_perc = 10 # This corresponds to 105% of MVC
desired_average_MVC_perc = 25 # This corresponds to 30% of MVC
perturbation_percentage_MVC_perc = 60 # This corresponds to 60% of MVC
base_percentage_MVC_perc = desired_average_MVC_perc
maximum_screen_MVC_percentage = 70
minimum_screen_MVC_percentage = 0
training_signal_duration = 30

# Parameters of the % screen
desired_sd_onscreen = desired_sd_MVC_perc*100/maximum_screen_MVC_percentage                                # This corresponds to desired_sd_true_perc of MVC
desired_average_onscreen = desired_average_MVC_perc*100/maximum_screen_MVC_percentage                      # This corresponds to desired_average_true_perc of MVC
perturbation_percentage_onscreen = perturbation_percentage_MVC_perc*100/maximum_screen_MVC_percentage      # This corresponds to perturbation_percentage_true_perc of MVC

# Parameters for training signals
Number_of_data_points_in_signal_pink = 65
Number_of_data_points_in_signal_white = 65
Number_of_data_points_in_signal_sine = 200
Number_of_data_points_in_signal_isotonic = 200
Number_of_cycles_in_sine_signal = 14
interpolation_factor = 5

# Parameter for the perturbation signals
num_sets = 1
time_signal_in_seconds = training_signal_duration * num_sets
percentage_of_time_of_perturbation_relative_to_signal_time = 0.5
time_perturbation_in_seconds = time_signal_in_seconds * percentage_of_time_of_perturbation_relative_to_signal_time
total_perturbation_trial_duration = time_signal_in_seconds + time_perturbation_in_seconds
print(f"The total duration of the set is {total_perturbation_trial_duration} seconds corresponding to {total_perturbation_trial_duration//60} minutes and {total_perturbation_trial_duration%60} seconds")
base_percentage_onscreen = desired_average_onscreen
base_part = np.full(1, base_percentage_onscreen)


# for i in range(1, 10):
#     # Pink training signals
#     pink_signal_before_interpolation = lb.pink_noise_signal_creation_using_FFT_method(Number_of_data_points_in_signal_pink, desired_sd_onscreen, desired_average_onscreen)
#     pink_signal = np.concatenate((pink_signal_before_interpolation, base_part), axis=0)
#     pink_signal = lb.signal_interpolation(pink_signal, interpolation_factor)
#
#     # White training signals
#     white_signal_before_interpolation = lb.white_noise_signal_creation_using_FFT_method(Number_of_data_points_in_signal_white, desired_sd_onscreen, desired_average_onscreen)
#     white_signal = np.concatenate((white_signal_before_interpolation, base_part), axis=0)
#     white_signal = lb.signal_interpolation(white_signal, interpolation_factor)
#
#     # Sine  training signals
#     sine_signal_before_interpolation = lb.sine_wave_signal_creation(Number_of_data_points_in_signal_sine, Number_of_cycles_in_sine_signal, desired_sd_onscreen, desired_average_onscreen)
#     sine_signal = lb.signal_interpolation(sine_signal_before_interpolation, interpolation_factor)
#
#     # Output of all signals
#     lb.outputs(white_signal_before_interpolation, pink_signal_before_interpolation, sine_signal_before_interpolation)

    # # Figure for training signals
    # time_pink = np.linspace(0, training_signal_duration, len(pink_signal))
    # time_white = np.linspace(0, training_signal_duration, len(white_signal))
    # time_sine = np.linspace(0, training_signal_duration, len(sine_signal))
    #
    # plt.scatter(time_pink, pink_signal, label='Pink', c='pink')
    # plt.scatter(time_white, white_signal, label='White', facecolors='white', edgecolors='black')
    # plt.scatter(time_sine, sine_signal, label='Sine', c='red')
    #
    # plt.plot(time_pink, pink_signal, lw=0.5, c='pink')
    # plt.plot(time_white, white_signal, lw=0.5, c='black')
    # plt.plot(time_sine, sine_signal, lw=0.5, c='red')
    #
    # plt.ylim(0, 100)
    # plt.ylabel("Screen (%)")
    # plt.xlabel("Time (s)")
    # plt.legend()
    #
    # ax = plt.gca()          # get current axes
    # ax2 = ax.twinx()        # create a twin y-axis sharing the same x-axis
    # ax2.set_ylim(0, maximum_screen_MVC_percentage)    # optional: set same limits, or different ones
    # ax2.set_ylabel("MVC (%)")  # optional label
    #
    # plt.show()

    # path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Signals\P6'
    # lb.create_txt_file(white_signal, f"White_signal_set_{i}", path)
    # lb.create_txt_file(pink_signal, f"Pink_signal_set_{i}", path)
    # lb.create_txt_file(sine_signal, f"Sine_signal_set_{i}", path)







# Pink training signal + perturbation
pink_signal_before_interpolation = lb.pink_noise_signal_creation_using_FFT_method(Number_of_data_points_in_signal_pink, desired_sd_onscreen, desired_average_onscreen)
pink_signal = np.concatenate((pink_signal_before_interpolation, base_part), axis=0)
pink_signal = lb.signal_interpolation(pink_signal, interpolation_factor)
Number_of_data_points_in_perturbation_pink = int(time_perturbation_in_seconds * len(pink_signal) / time_signal_in_seconds)
pink_perturbation_signal = np.full(Number_of_data_points_in_perturbation_pink, perturbation_percentage_onscreen)
pink_signal_with_perturbation = np.concatenate((pink_signal, base_part, pink_perturbation_signal), axis=0)

# White training signal + perturbation
white_signal_before_interpolation = lb.white_noise_signal_creation_using_FFT_method(Number_of_data_points_in_signal_white, desired_sd_onscreen, desired_average_onscreen)
white_signal = np.concatenate((white_signal_before_interpolation, base_part), axis=0)
white_signal = lb.signal_interpolation(white_signal, interpolation_factor)
Number_of_data_points_in_perturbation_white = int(time_perturbation_in_seconds * len(white_signal) / time_signal_in_seconds)
white_perturbation_signal = np.full(Number_of_data_points_in_perturbation_white, perturbation_percentage_onscreen)
white_signal_with_perturbation = np.concatenate((white_signal, white_perturbation_signal), axis=0)

# Sine training signal + perturbation
sine_signal_before_interpolation = lb.sine_wave_signal_creation(Number_of_data_points_in_signal_sine, Number_of_cycles_in_sine_signal, desired_sd_onscreen, desired_average_onscreen)
sine_signal = lb.signal_interpolation(sine_signal_before_interpolation, interpolation_factor)
Number_of_data_points_in_perturbation_sine = int(time_perturbation_in_seconds * len(sine_signal) / time_signal_in_seconds)
sine_perturbation_signal = np.full(Number_of_data_points_in_perturbation_sine, perturbation_percentage_onscreen)
sine_signal_with_perturbation = np.concatenate((sine_signal, sine_perturbation_signal), axis=0)

# Figure for training signals
time_pink = np.linspace(0, total_perturbation_trial_duration, len(pink_signal_with_perturbation))
time_white = np.linspace(0, total_perturbation_trial_duration, len(white_signal_with_perturbation))
time_sine = np.linspace(0, total_perturbation_trial_duration, len(sine_signal_with_perturbation))

plt.scatter(time_pink, pink_signal_with_perturbation, label='Pink', c='pink')
# plt.scatter(time_white, white_signal_with_perturbation, label='White', facecolors='white', edgecolors='black')
# plt.scatter(time_sine, sine_signal_with_perturbation, label='Sine', c='red')

plt.plot(time_pink, pink_signal_with_perturbation, lw=0.5, c='pink')
# plt.plot(time_white, white_signal_with_perturbation, lw=0.5, c='black')
# plt.plot(time_sine, sine_signal_with_perturbation, lw=0.5, c='red')

plt.ylim(0, 100)
plt.ylabel("Screen (%)")
plt.xlabel("Time (s)")
plt.legend()

ax = plt.gca()          # get current axes
ax2 = ax.twinx()        # create a twin y-axis sharing the same x-axis
ax2.set_ylim(0, maximum_screen_MVC_percentage)    # optional: set same limits, or different ones
ax2.set_ylabel("MVC (%)")  # optional label

plt.show()

path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Signals\P6'
# lb.create_txt_file(pink_signal_with_perturbation, f"Pink_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_interp_{interpolation_factor}", path)
# lb.create_txt_file(white_signal_with_perturbation, f"White_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_interp_{interpolation_factor}", path)
# lb.create_txt_file(sine_signal_with_perturbation, f"Sine_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_interp_{interpolation_factor}", path)




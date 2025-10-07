import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
import random

# Parameters of the % MVC
desired_sd_MVC_perc = 5 # This corresponds to 5% of MVC
desired_average_MVC_perc = 20 # This corresponds to 20% of MVC
perturbation_percentage_MVC_perc = 70 # This corresponds to 70% of MVC
base_percentage_MVC_perc = desired_average_MVC_perc
maximum_screen_MVC_percentage = 80
minimum_screen_MVC_percentage = 0

# Parameters of the % screen
desired_sd_onscreen = desired_sd_MVC_perc*100/maximum_screen_MVC_percentage                                # This corresponds to desired_sd_true_perc of MVC
desired_average_onscreen = desired_average_MVC_perc*100/maximum_screen_MVC_percentage                      # This corresponds to desired_average_true_perc of MVC
perturbation_percentage_onscreen = perturbation_percentage_MVC_perc*100/maximum_screen_MVC_percentage      # This corresponds to perturbation_percentage_true_perc of MVC
base_percentage_onscreen = desired_average_onscreen
young_trails_seconds = 30

# Parameters of signals
num_sets = 7
time_signal_in_seconds = 30 * num_sets
extrapolation_factor = 10
percentage_of_time_of_perturbation_relative_to_signal_time = 0.2
time_perturbation_in_seconds = time_signal_in_seconds * percentage_of_time_of_perturbation_relative_to_signal_time
Number_of_data_points_in_signal_pink = 65 * num_sets
Number_of_data_points_in_signal_white = 65 * num_sets
Number_of_data_points_in_signal_sine = 200 * num_sets
Number_of_data_points_in_signal_isotonic = 200 * num_sets
Number_of_cycles_in_sine_signal = 14*num_sets


base_part = np.full(1, base_percentage_onscreen)

# Pink
pink_signal = lb.fgn_sim(Number_of_data_points_in_signal_pink, 0.99)
pink_signal = lb.z_transform(pink_signal, desired_sd_onscreen, desired_average_onscreen)
pink_signal = np.concatenate((pink_signal, base_part), axis=0)
pink_signal = lb.signal_extrapolation(pink_signal, extrapolation_factor)
Number_of_data_points_in_perturbation_pink = int(time_perturbation_in_seconds * len(pink_signal) / time_signal_in_seconds)
pink_perturbation_signal = np.full(Number_of_data_points_in_perturbation_pink, perturbation_percentage_onscreen)
pink_signal_with_pert = np.concatenate((pink_signal, pink_perturbation_signal), axis=0)

# White
white_signal = lb.white_noise_signal_creation_using_FFT_method(Number_of_data_points_in_signal_white, desired_sd_onscreen, desired_average_onscreen)
white_signal = lb.z_transform(white_signal, desired_sd_onscreen, desired_average_onscreen)
white_signal = np.concatenate((white_signal, base_part), axis=0)
white_signal = lb.signal_extrapolation(white_signal, extrapolation_factor)
Number_of_data_points_in_perturbation_white = int(time_perturbation_in_seconds * len(white_signal) / time_signal_in_seconds)
white_perturbation_signal = np.full(Number_of_data_points_in_perturbation_white, perturbation_percentage_onscreen)
white_signal_with_pert = np.concatenate((white_signal, white_perturbation_signal), axis=0)

# Sine
sine_signal = lb.sine_wave_signal_creation(Number_of_data_points_in_signal_sine, Number_of_cycles_in_sine_signal, desired_sd_onscreen, desired_average_onscreen)
sine_signal = lb.z_transform(sine_signal, desired_sd_onscreen, desired_average_onscreen)
sine_signal = np.concatenate((sine_signal, base_part), axis=0)
sine_signal = lb.signal_extrapolation(sine_signal, extrapolation_factor)
Number_of_data_points_in_perturbation_sine = int(time_perturbation_in_seconds * len(sine_signal) / time_signal_in_seconds)
sine_perturbation_signal = np.full(Number_of_data_points_in_perturbation_sine, perturbation_percentage_onscreen)
sine_signal_with_pert = np.concatenate((sine_signal, sine_perturbation_signal), axis=0)

# Isotonic
isotonic_signal = np.full(Number_of_data_points_in_signal_isotonic, desired_average_onscreen)
isotonic_signal = lb.signal_extrapolation(isotonic_signal, extrapolation_factor)
Number_of_data_points_in_perturbation_isotonic = int(time_perturbation_in_seconds * len(isotonic_signal) / time_signal_in_seconds)
isotonic_perturbation_signal = np.full(Number_of_data_points_in_perturbation_isotonic, perturbation_percentage_onscreen)
isotonic_signal_with_pert = np.concatenate((isotonic_signal, isotonic_perturbation_signal), axis=0)

lb.outputs(white_signal, pink_signal, sine_signal)


# Time for the plots
time = np.linspace(0, time_signal_in_seconds + time_perturbation_in_seconds, len(pink_signal_with_pert))
time_sine = np.linspace(0, time_signal_in_seconds + time_perturbation_in_seconds, len(sine_signal_with_pert))
time_isotonic = np.linspace(0, time_signal_in_seconds + time_perturbation_in_seconds, len(isotonic_signal_with_pert))





plt.scatter(time, pink_signal_with_pert, label='Pink', c='pink')
plt.scatter(time, white_signal_with_pert, label='White', facecolors='white', edgecolors='black')
plt.scatter(time_sine, sine_signal_with_pert, label='Sine', c='red')
plt.scatter(time_isotonic, isotonic_signal_with_pert, label='Isotonic', c='blue')

plt.plot(time, pink_signal_with_pert, lw=0.5, c='pink')
plt.plot(time, white_signal_with_pert, lw=0.5, c='black')
plt.plot(time_sine, sine_signal_with_pert, lw=0.5, c='red')
plt.plot(time_isotonic, isotonic_signal_with_pert, lw=0.5, c='blue')

plt.ylim(0, 100)
plt.ylabel("Percentage of the screen (%)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

directory_to_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Signals\P3'
lb.create_txt_file(pink_signal_with_pert, f"Pink_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_ext_{extrapolation_factor}", directory_to_save)
lb.create_txt_file(white_signal_with_pert, f"White_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_ext_{extrapolation_factor}", directory_to_save)
lb.create_txt_file(sine_signal_with_pert, f"Sine_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_ext_{extrapolation_factor}", directory_to_save)
lb.create_txt_file(isotonic_signal_with_pert, f"Isotonic_average_{desired_average_MVC_perc}_sd_{desired_sd_MVC_perc}_pert_{perturbation_percentage_MVC_perc}_screenmax_{maximum_screen_MVC_percentage}_ext_{extrapolation_factor}", directory_to_save)




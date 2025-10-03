import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
import random


desired_sd_true_perc = 5 # This corresponds to 5% of MVC
desired_average_true_perc = 20 # This corresponds to 20% of MVC
perturbation_percentage_true_perc = 70 # This corresponds to 70% of MVC
base_percentage_true_perc = desired_average_true_perc
maximum_screen_MVC_percentage = 80

desired_sd_onscreen = desired_sd_true_perc*100/maximum_screen_MVC_percentage                                # This corresponds to desired_sd_true_perc of MVC
desired_average_onscreen = desired_average_true_perc*100/maximum_screen_MVC_percentage                      # This corresponds to desired_average_true_perc of MVC
perturbation_percentage_onscreen = perturbation_percentage_true_perc*100/maximum_screen_MVC_percentage      # This corresponds to perturbation_percentage_true_perc of MVC
base_percentage_onscreen = desired_average_onscreen

Number_of_values_in_signal = 420 # This corresponds to 3.5 minutes of targets before perturbation


perturbation_part = np.full(100, perturbation_percentage_onscreen)
base_part = np.full(1, base_percentage_onscreen)

# Pink
pink_signal = lb.fgn_sim(Number_of_values_in_signal, 0.99)
pink_signal = lb.z_transform(pink_signal, desired_sd_onscreen, desired_average_onscreen)
pink_signal_with_pert = np.concatenate((pink_signal, base_part, perturbation_part), axis=0)

# White
white_signal = lb.white_noise_signal_creation_using_FFT_method(Number_of_values_in_signal, desired_sd_onscreen, desired_average_onscreen)
white_signal = lb.z_transform(white_signal, desired_sd_onscreen, desired_average_onscreen)
white_signal_with_pert = np.concatenate((white_signal, base_part, perturbation_part), axis=0)

# Sine
sine_perturbation_part = np.full(1000, perturbation_percentage_onscreen)
sine_signal = lb.sine_wave_signal_creation(Number_of_values_in_signal*10, 105, desired_sd_onscreen, desired_average_onscreen)
sine_signal = lb.z_transform(sine_signal, desired_sd_onscreen, desired_average_onscreen)
sine_signal_with_pert = np.concatenate((sine_signal, base_part, sine_perturbation_part), axis=0)

# Isotonic
isotonic_signal = np.full(420, desired_average_onscreen)
isotonic_signal_with_pert = np.concatenate((base_part, isotonic_signal, perturbation_part), axis=0)

lb.outputs(white_signal, pink_signal, sine_signal)

# Figure to see the signals
time = np.linspace(0, len(pink_signal_with_pert)/2, len(pink_signal_with_pert))
time_sine = np.linspace(0, len(pink_signal_with_pert)/2, len(sine_signal_with_pert))
print(len(sine_signal_with_pert))
print(len(pink_signal_with_pert))
print(len(time))
print(len(time_sine))
print(time[:-1])
print(time_sine[:-1])


plt.scatter(time[:-1], pink_signal_with_pert[1:], label='Pink', c='pink')
plt.scatter(time[:-1], white_signal_with_pert[1:], label='White', facecolors='white', edgecolors='black')
plt.scatter(time_sine[:-1], sine_signal_with_pert[:-1], label='Sine', c='red')
plt.scatter(time[:-1], isotonic_signal_with_pert[1:], label='Isotonic', c='blue')

plt.plot(time[:-1], pink_signal_with_pert[1:], lw=0.5, c='pink')
plt.plot(time[:-1], white_signal_with_pert[1:], lw=0.5, c='black')
plt.plot(time_sine[:-1], sine_signal_with_pert[:-1], lw=0.5, c='red')
plt.plot(time[:-1], isotonic_signal_with_pert[1:], lw=0.5, c='blue')

plt.ylim(0, 100)
plt.legend()
plt.show()




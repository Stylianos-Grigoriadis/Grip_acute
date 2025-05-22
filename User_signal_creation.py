import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
import random

def creation_signals(desired_average, desired_sd, perturbation_percentage, Number_of_values_in_signal):
    for i in range(1,6):
        print(i)

        # Creation of signals already z transformed and checked for having no zeros
        # desired_sd = 10
        # desired_average = 75
        # perturbation_percentage = 25
        base_percentage = desired_average

        pink_signal = lb.pink_noise_signal_creation_using_FFT_method(Number_of_values_in_signal, desired_sd, desired_average)
        white_signal = lb.white_noise_signal_creation_using_FFT_method(Number_of_values_in_signal, desired_sd, desired_average)
        sine_signal = lb.sine_wave_signal_creation(Number_of_values_in_signal, 10, desired_sd, desired_average)
        invariant_signal = np.full(Number_of_values_in_signal,desired_average)

        # Creation of perturbation signal (we use 1 value for pre and 1 value for post perturbation

        perturbation_part = np.full(50,perturbation_percentage)
        base_part = np.full(1,base_percentage)

        # Merge the signal with the perturbation to create the last signal
        pink_signal = np.concatenate((pink_signal, base_part, perturbation_part), axis=0)
        white_signal = np.concatenate((white_signal, base_part, perturbation_part), axis=0)
        sine_signal = np.concatenate((sine_signal, base_part, perturbation_part), axis=0)
        invariant_signal = np.concatenate((invariant_signal, base_part, perturbation_part), axis=0)

        # Print the resulted total load, average, and std
        lb.outputs(white_signal, pink_signal, sine_signal)

        # Figure to see the signals
        # time = np.arange(0, len(pink_signal),1)
        #
        # plt.scatter(time, pink_signal,label='pink_signal')
        # plt.scatter(time, white_signal,label='white_signal')
        # plt.scatter(time, sine_signal,label='sine_signal')
        # plt.scatter(time, invariant_signal,label='invariant_signal')
        #
        # plt.plot(time, pink_signal,lw=0.5)
        # plt.plot(time, white_signal,lw=0.5)
        # plt.plot(time, sine_signal,lw=0.5)
        # plt.plot(time, invariant_signal,lw=0.5)
        #
        # plt.legend()
        # plt.show()


        directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute\Data\Signals\P1'
        os.chdir(directory)

        pink_name = f'pink_{desired_average}_to_{perturbation_percentage}_sd_{desired_sd}_T_{i}'
        white_name = f'white_{desired_average}_to_{perturbation_percentage}_sd_{desired_sd}_T_{i}'
        sine_name = f'sine_{desired_average}_to_{perturbation_percentage}_sd_{desired_sd}_T_{i}'
        invariant_name = f'invariant_{desired_average}_to_{perturbation_percentage}_sd_{desired_sd}_T_{i}'

        # lb.create_txt_file(pink_signal, pink_name, directory)
        # lb.create_txt_file(white_signal, white_name, directory)
        # lb.create_txt_file(sine_signal, sine_name, directory)
        # lb.create_txt_file(invariant_signal, invariant_name, directory)




# def trials_order_randomization():
#     trials = [
#         "pink_25_to_50_sd_1_T_1", "pink_25_to_50_sd_1_T_2", "pink_25_to_50_sd_1_T_3", "pink_25_to_50_sd_1_T_4",
#         "pink_25_to_50_sd_1_T_5",
#         "pink_25_to_75_sd_1_T_1", "pink_25_to_75_sd_1_T_2", "pink_25_to_75_sd_1_T_3", "pink_25_to_75_sd_1_T_4",
#         "pink_25_to_75_sd_1_T_5",
#         "pink_75_to_50_sd_1_T_1", "pink_75_to_50_sd_1_T_2", "pink_75_to_50_sd_1_T_3", "pink_75_to_50_sd_1_T_4",
#         "pink_75_to_50_sd_1_T_5",
#         "pink_75_to_25_sd_1_T_1", "pink_75_to_25_sd_1_T_2", "pink_75_to_25_sd_1_T_3", "pink_75_to_25_sd_1_T_4",
#         "pink_75_to_25_sd_1_T_5",
#         "white_25_to_50_sd_1_T_1", "white_25_to_50_sd_1_T_2", "white_25_to_50_sd_1_T_3", "white_25_to_50_sd_1_T_4",
#         "white_25_to_50_sd_1_T_5",
#         "white_25_to_75_sd_1_T_1", "white_25_to_75_sd_1_T_2", "white_25_to_75_sd_1_T_3", "white_25_to_75_sd_1_T_4",
#         "white_25_to_75_sd_1_T_5",
#         "white_75_to_50_sd_1_T_1", "white_75_to_50_sd_1_T_2", "white_75_to_50_sd_1_T_3", "white_75_to_50_sd_1_T_4",
#         "white_75_to_50_sd_1_T_5",
#         "white_75_to_25_sd_1_T_1", "white_75_to_25_sd_1_T_2", "white_75_to_25_sd_1_T_3", "white_75_to_25_sd_1_T_4",
#         "white_75_to_25_sd_1_T_5",
#         "iso_25_to_50_sd_1_T_1", "iso_25_to_50_sd_1_T_2", "iso_25_to_50_sd_1_T_3", "iso_25_to_50_sd_1_T_4",
#         "iso_25_to_50_sd_1_T_5",
#         "iso_25_to_75_sd_1_T_1", "iso_25_to_75_sd_1_T_2", "iso_25_to_75_sd_1_T_3", "iso_25_to_75_sd_1_T_4",
#         "iso_25_to_75_sd_1_T_5",
#         "iso_75_to_50_sd_1_T_1", "iso_75_to_50_sd_1_T_2", "iso_75_to_50_sd_1_T_3", "iso_75_to_50_sd_1_T_4",
#         "iso_75_to_50_sd_1_T_5",
#         "iso_75_to_25_sd_1_T_1", "iso_75_to_25_sd_1_T_2", "iso_75_to_25_sd_1_T_3", "iso_75_to_25_sd_1_T_4",
#         "iso_75_to_25_sd_1_T_5",
#         "sine_25_to_50_sd_1_T_1", "sine_25_to_50_sd_1_T_2", "sine_25_to_50_sd_1_T_3", "sine_25_to_50_sd_1_T_4",
#         "sine_25_to_50_sd_1_T_5",
#         "sine_25_to_75_sd_1_T_1", "sine_25_to_75_sd_1_T_2", "sine_25_to_75_sd_1_T_3", "sine_25_to_75_sd_1_T_4",
#         "sine_25_to_75_sd_1_T_5",
#         "sine_75_to_50_sd_1_T_1", "sine_75_to_50_sd_1_T_2", "sine_75_to_50_sd_1_T_3", "sine_75_to_50_sd_1_T_4",
#         "sine_75_to_50_sd_1_T_5",
#         "sine_75_to_25_sd_1_T_1", "sine_75_to_25_sd_1_T_2", "sine_75_to_25_sd_1_T_3", "sine_75_to_25_sd_1_T_4",
#         "sine_75_to_25_sd_1_T_5"
#     ]
#     random.shuffle(trials)
#     randomized_trials = ', '.join(trials)
#     print(randomized_trials)

creation_signals(desired_average=75, desired_sd=10, perturbation_percentage=50, Number_of_values_in_signal=100)
creation_signals(desired_average=75, desired_sd=10, perturbation_percentage=25, Number_of_values_in_signal=100)
creation_signals(desired_average=25, desired_sd=10, perturbation_percentage=50, Number_of_values_in_signal=100)
creation_signals(desired_average=25, desired_sd=10, perturbation_percentage=75, Number_of_values_in_signal=100)





# directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute\Data\Signals\P1'
# files = glob.glob(os.path.join(directory_path, "*"))
# for file in files:
#     print(file)
# trials_order_randomization()
#
# pink_25_to_50_sd_1_T_1
# pink_25_to_50_sd_1_T_2
# pink_25_to_50_sd_1_T_3
# pink_25_to_50_sd_1_T_4
# pink_25_to_50_sd_1_T_5
# pink_25_to_75_sd_1_T_1
# pink_25_to_75_sd_1_T_2
# pink_25_to_75_sd_1_T_3
# pink_25_to_75_sd_1_T_4
# pink_25_to_75_sd_1_T_5
# pink_75_to_50_sd_1_T_1
# pink_75_to_50_sd_1_T_2
# pink_75_to_50_sd_1_T_3
# pink_75_to_50_sd_1_T_4
# pink_75_to_50_sd_1_T_5
# pink_75_to_25_sd_1_T_1
# pink_75_to_25_sd_1_T_2
# pink_75_to_25_sd_1_T_3
# pink_75_to_25_sd_1_T_4
# pink_75_to_25_sd_1_T_5
# white_25_to_50_sd_1_T_1
# white_25_to_50_sd_1_T_2
# white_25_to_50_sd_1_T_3
# white_25_to_50_sd_1_T_4
# white_25_to_50_sd_1_T_5
# white_25_to_75_sd_1_T_1
# white_25_to_75_sd_1_T_2
# white_25_to_75_sd_1_T_3
# white_25_to_75_sd_1_T_4
# white_25_to_75_sd_1_T_5
# white_75_to_50_sd_1_T_1
# white_75_to_50_sd_1_T_2
# white_75_to_50_sd_1_T_3
# white_75_to_50_sd_1_T_4
# white_75_to_50_sd_1_T_5
# white_75_to_25_sd_1_T_1
# white_75_to_25_sd_1_T_2
# white_75_to_25_sd_1_T_3
# white_75_to_25_sd_1_T_4
# white_75_to_25_sd_1_T_5
# iso_25_to_50_sd_1_T_1
# iso_25_to_50_sd_1_T_2
# iso_25_to_50_sd_1_T_3
# iso_25_to_50_sd_1_T_4
# iso_25_to_50_sd_1_T_5
# iso_25_to_75_sd_1_T_1
# iso_25_to_75_sd_1_T_2
# iso_25_to_75_sd_1_T_3
# iso_25_to_75_sd_1_T_4
# iso_25_to_75_sd_1_T_5
# iso_75_to_50_sd_1_T_1
# iso_75_to_50_sd_1_T_2
# iso_75_to_50_sd_1_T_3
# iso_75_to_50_sd_1_T_4
# iso_75_to_50_sd_1_T_5
# iso_75_to_25_sd_1_T_1
# iso_75_to_25_sd_1_T_2
# iso_75_to_25_sd_1_T_3
# iso_75_to_25_sd_1_T_4
# iso_75_to_25_sd_1_T_5
# sine_25_to_50_sd_1_T_1
# sine_25_to_50_sd_1_T_2
# sine_25_to_50_sd_1_T_3
# sine_25_to_50_sd_1_T_4
# sine_25_to_50_sd_1_T_5
# sine_25_to_75_sd_1_T_1
# sine_25_to_75_sd_1_T_2
# sine_25_to_75_sd_1_T_3
# sine_25_to_75_sd_1_T_4
# sine_25_to_75_sd_1_T_5
# sine_75_to_50_sd_1_T_1
# sine_75_to_50_sd_1_T_2
# sine_75_to_50_sd_1_T_3
# sine_75_to_50_sd_1_T_4
# sine_75_to_50_sd_1_T_5
# sine_75_to_25_sd_1_T_1
# sine_75_to_25_sd_1_T_2
# sine_75_to_25_sd_1_T_3
# sine_75_to_25_sd_1_T_4
# sine_75_to_25_sd_1_T_5

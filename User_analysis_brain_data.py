import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os

directory_hemoglobin = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Brain data'
parts = directory_hemoglobin.split(os.sep)
ID = parts[-2]
list_training_sets, fs = lb.artinis_read_file_10_sets(directory_hemoglobin, ID)
for i in range(len(list_training_sets)):
    training_set = list_training_sets[i]
    # I will check it out but hypothetically the Rx1 and Rx3 are the long distance receiver
    # I will check it out but hypothetically the 9322 is left side and the 9323 is right side
    print(f'Set is {i + 1}')

    time = training_set['Time'].to_numpy()
    time = time - time[0]

    left_Rx1_TSI_perc = training_set['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'].to_numpy()
    left_RX1_TSI_Fit_Factor = training_set['[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor'].to_numpy()
    left_Rx1_Tx1_O2Hb = training_set['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = training_set['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = training_set['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
    left_Rx2_Tx1_O2Hb = training_set['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
    left_Rx2_Tx2_O2Hb = training_set['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
    left_Rx2_Tx3_O2Hb = training_set['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
    left_Rx1_Tx1_HHb = training_set['[9322] Rx1 - Tx1  HHb'].to_numpy()
    left_Rx1_Tx2_HHb = training_set['[9322] Rx1 - Tx2  HHb'].to_numpy()
    left_Rx1_Tx3_HHb = training_set['[9322] Rx1 - Tx3  HHb'].to_numpy()
    left_Rx2_Tx1_HHb = training_set['[9322] Rx2 - Tx1  HHb'].to_numpy()
    left_Rx2_Tx2_HHb = training_set['[9322] Rx2 - Tx2  HHb'].to_numpy()
    left_Rx2_Tx3_HHb = training_set['[9322] Rx2 - Tx3  HHb'].to_numpy()

    right_Rx3_TSI_perc = training_set['[9323] Rx3 - Tx4,Tx5,Tx6  TSI%'].to_numpy()
    right_RX3_TSI_Fit_Factor = training_set['[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor'].to_numpy()
    right_Rx3_Tx4_O2Hb = training_set['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = training_set['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = training_set['[9323] Rx3 - Tx6  O2Hb'].to_numpy()
    right_Rx4_Tx4_O2Hb = training_set['[9323] Rx4 - Tx4  O2Hb'].to_numpy()
    right_Rx4_Tx5_O2Hb = training_set['[9323] Rx4 - Tx5  O2Hb'].to_numpy()
    right_Rx4_Tx6_O2Hb = training_set['[9323] Rx4 - Tx6  O2Hb'].to_numpy()
    right_Rx3_Tx4_HHb = training_set['[9323] Rx3 - Tx4  HHb'].to_numpy()
    right_Rx3_Tx5_HHb = training_set['[9323] Rx3 - Tx5  HHb'].to_numpy()
    right_Rx3_Tx6_HHb = training_set['[9323] Rx3 - Tx6  HHb'].to_numpy()
    right_Rx4_Tx4_HHb = training_set['[9323] Rx4 - Tx4  HHb'].to_numpy()
    right_Rx4_Tx5_HHb = training_set['[9323] Rx4 - Tx5  HHb'].to_numpy()
    right_Rx4_Tx6_HHb = training_set['[9323] Rx4 - Tx6  HHb'].to_numpy()

    left_Rx1_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)

    left_short_channels_O2Hb = [left_Rx2_Tx1_O2Hb, left_Rx2_Tx2_O2Hb, left_Rx2_Tx3_O2Hb]
    left_short_channels_HHb = [left_Rx2_Tx1_HHb, left_Rx2_Tx2_HHb, left_Rx2_Tx3_HHb]
    right_short_channels_O2Hb = [right_Rx4_Tx4_O2Hb, right_Rx4_Tx5_O2Hb, right_Rx4_Tx6_O2Hb]
    right_short_channels_HHb = [right_Rx4_Tx4_HHb, right_Rx4_Tx5_HHb, right_Rx4_Tx6_HHb]

    left_O2Hb_PC_list, left_O2Hb_PC_explained_variance  = lb.Principal_component_analysis(left_short_channels_O2Hb, plot=False)
    left_HHb_PC_list, left_HHb_PC_explained_variance  = lb.Principal_component_analysis(left_short_channels_HHb, plot=False)
    right_O2Hb_PC_list, right_O2Hb_PC_explained_variance  = lb.Principal_component_analysis(right_short_channels_O2Hb, plot=False)
    right_HHb_PC_list, right_HHb_PC_explained_variance  = lb.Principal_component_analysis(right_short_channels_HHb, plot=False)
    # print(f'Variation explain from left O2Hb PC1 is {left_O2Hb_PC_explained_variance[0]}')
    # print(f'Variation explain from left HHb PC1 is {left_HHb_PC_explained_variance[0]}')
    # print(f'Variation explain from right O2Hb PC1 is {right_O2Hb_PC_explained_variance[0]}')
    # print(f'Variation explain from right HHb PC1 is {right_HHb_PC_explained_variance[0]}')

    PC1_left_O2Hb = left_O2Hb_PC_list[0]
    PC1_left_HHb = left_HHb_PC_list[0]
    PC1_right_O2Hb = right_O2Hb_PC_list[0]
    PC1_right_HHb = right_HHb_PC_list[0]

    start_time = 10         # 10th second the trial begun
    binary_rest_task = lb.task_array_binary(time, start_time)
    HFR = lb.make_hrf(fs)
    # plt.plot(HFR)
    # plt.show()

    task_reg_z, binary_rest_task, task_reg = lb.build_task_regressor(binary_rest_task, HFR)

    x_HFR = np.linspace(0,40,len(HFR))
    x_task_reg_z = np.linspace(0,40,len(task_reg_z))
    x_task_reg = np.linspace(0,40,len(task_reg))
    #
    # plt.plot(x_HFR, HFR, label='HFR')
    # plt.plot(x_task_reg_z, task_reg_z, label='task_reg_z')
    # plt.plot(x_task_reg, task_reg, label='task_reg')
    # plt.legend()
    # plt.show()

    # fig, ax1 = plt.subplots()
    #
    # # Primary axis
    # ax1.plot(x_HFR, HFR, label='HFR')
    # ax1.plot(x_task_reg_z, task_reg_z, label='task_reg_z')
    # ax1.set_ylabel("Primary axis")
    # ax1.legend(loc="upper left")
    #
    # # Secondary axis
    # ax2 = ax1.twinx()
    # ax2.plot(x_task_reg, task_reg, color="orange", label='task_reg')
    # ax2.set_ylabel("Secondary axis")
    #
    # # Separate legend for secondary axis
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    #
    # plt.show()
    beta_task, betas, y_hat, cleaned, y_hat_R2, corralation_between_actual_data_and_y_hat, full_R2, partial_R2_task, partial_R2_pc1, partial_R2_short = lb.run_glm_simple(left_Rx1_Tx1_O2Hb, task_reg_z, PC1_left_O2Hb, left_Rx2_Tx1_O2Hb)
    print(f'corralation_between_actual_data_and_y_hat is {corralation_between_actual_data_and_y_hat}')
    print(f'full_R2 is {full_R2}')
    print(f'partial_R2_task is {partial_R2_task}')
    print(f'partial_R2_pc1 is {partial_R2_pc1}')
    print(f'partial_R2_short is {partial_R2_short}')

    plt.plot(y_hat, label='y_hat')
    plt.plot(cleaned, label='cleaned')
    plt.plot(left_Rx1_Tx1_O2Hb, label='left_Rx1_Tx1_O2Hb')
    plt.legend()
    plt.show()







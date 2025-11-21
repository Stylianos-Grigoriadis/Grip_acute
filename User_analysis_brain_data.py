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
list_training_sets, fs = lb.artinis_read_file_10_sets(directory_hemoglobin, name)
for i in range(len(list_training_sets)):
    training_set = list_training_sets[i]
    # I will check it out but hypothetically the Rx1 and Rx3 are the long distance receiver
    # I will check it out but hypothetically the 9322 is left side and the 9323 is right side
    print(training_set.columns)
    print(f'Set is {i + 1}')

    time = training_set['Time'].to_numpy()

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

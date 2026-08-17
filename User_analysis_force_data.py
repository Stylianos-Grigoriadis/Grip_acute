import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import os

directory = r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen"

list_ID = []
list_group_ID = []
for folder in os.listdir(directory):
    ID = str(folder)
    list_ID.append(ID)
    print(ID)
    group = folder.split("_")[0]
    list_group_ID.append(group)
    print(group)

    grip_folder = os.path.join(directory, folder, "Grip data")
    os.chdir(grip_folder)

    Training_1 = pd.read_csv("Training_1.csv", skiprows=2)
    Training_2 = pd.read_csv("Training_2.csv", skiprows=2)
    Training_3 = pd.read_csv("Training_3.csv", skiprows=2)
    Training_4 = pd.read_csv("Training_4.csv", skiprows=2)
    Training_5 = pd.read_csv("Training_5.csv", skiprows=2)
    Training_6 = pd.read_csv("Training_6.csv", skiprows=2)
    Training_7 = pd.read_csv("Training_7.csv", skiprows=2)
    Training_8 = pd.read_csv("Training_8.csv", skiprows=2)
    Training_9 = pd.read_csv("Training_9.csv", skiprows=2)
    Training_10 = pd.read_csv("Training_10.csv", skiprows=2)
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
    Isometric_high = pd.read_csv("Isometric_high.csv", skiprows=2)
    Isometric_low = pd.read_csv("Isometric_low.csv", skiprows=2)

    #############################
    ##### Residual Analysis #####
    #############################
    cutoff_frequencies = np.arange(1, 50)
    fs = 100
    signals = [Training_1, Training_2, Training_3, Training_4, Training_5, Training_6, Training_7, Training_8, Training_9, Training_10, Pre_Pert_down_1, Pre_Pert_down_2, Pre_Pert_down_3, Pre_Pert_up_1, Pre_Pert_up_2, Pre_Pert_up_3, Post_Pert_down_1, Post_Pert_down_2, Post_Pert_down_3, Post_Pert_up_1, Post_Pert_up_2, Post_Pert_up_3, Isometric_high, Isometric_low]
    signal_names = ["Training_1", "Training_2", "Training_3", "Training_4", "Training_5", "Training_6", "Training_7", "Training_8", "Training_9", "Training_10", "Pre_Pert_down_1", "Pre_Pert_down_2", "Pre_Pert_down_3", "Pre_Pert_up_1", "Pre_Pert_up_2", "Pre_Pert_up_3", "Post_Pert_down_1", "Post_Pert_down_2", "Post_Pert_down_3", "Post_Pert_up_1", "Post_Pert_up_2", "Post_Pert_up_3", "Isometric_high", "Isometric_low"]
    list_residual_FC = []
    for signal in signals:
        signal_Fc = lib.Residual_analysis(signal["Performance"], cutoff_frequencies, fs, number_of_fit_points=10, plot=False, save=None)
        list_residual_FC.append(signal_Fc)
    plt.plot(list_residual_FC, marker="o")

    plt.xticks(
        range(len(signal_names)),
        signal_names,
        rotation=90
    )

    plt.ylabel("Optimal Cutoff frequency (Hz)")
    plt.tight_layout()
    save_name = ID + ".png"
    plt.savefig(
        os.path.join(r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Figures\Residual plots Grip data', save_name),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

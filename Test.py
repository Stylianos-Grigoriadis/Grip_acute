import numpy as np
import matplotlib.pyplot as plt
import lib
import pandas as pd
import Lib_grip as lb
import polars as pl


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Signals\Pilot'
name = r'Stylianos_test_11-24-2025'
data = pl.read_excel(f"{directory}\\{name}.xlsx", infer_schema_length=None)
data = data.rename({data.columns[1]: "Unnamed: 1"})
data = data.slice(2)
sampling_frequency = float(data['Unnamed: 1'][0])
total_samples = float(data['Unnamed: 1'][2])
step = 1 / sampling_frequency
time = np.arange(0, total_samples * step, step)

column_names = ['Sample number', '[9322] Rx1 - Tx1,Tx2,Tx3  TSI%', '[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor',
                    '[9323] Rx3 - Tx4,Tx5,Tx6  TSI%', '[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor',
                    '[9322] Rx1 - Tx1  O2Hb', '[9322] Rx1 - Tx1  HHb', '[9322] Rx1 - Tx2  O2Hb',
                    '[9322] Rx1 - Tx2  HHb', '[9322] Rx1 - Tx3  O2Hb', '[9322] Rx1 - Tx3  HHb',
                    '[9322] Rx2 - Tx1  O2Hb', '[9322] Rx2 - Tx1  HHb', '[9322] Rx2 - Tx2  O2Hb',
                    '[9322] Rx2 - Tx2  HHb', '[9322] Rx2 - Tx3  O2Hb', '[9322] Rx2 - Tx3  HHb',
                    '[9323] Rx3 - Tx4  O2Hb', '[9323] Rx3 - Tx4  HHb', '[9323] Rx3 - Tx5  O2Hb',
                    '[9323] Rx3 - Tx5  HHb', '[9323] Rx3 - Tx6  O2Hb', '[9323] Rx3 - Tx6  HHb',
                    '[9323] Rx4 - Tx4  O2Hb', '[9323] Rx4 - Tx4  HHb', '[9323] Rx4 - Tx5  O2Hb',
                    '[9323] Rx4 - Tx5  HHb', '[9323] Rx4 - Tx6  O2Hb', '[9323] Rx4 - Tx6  HHb', 'Event', 'Event text']
data = data.slice(63)

data = data.rename(dict(zip(data.columns, column_names)))
data = data.with_columns(
    pl.Series("Time", time)
)
cols = data.columns
new_order = [cols[0], "Time"] + cols[1:-1]  # move Time after the first column
data = data.select(new_order)

data = data.with_columns(
    pl.col("Event").cast(pl.Utf8)
)
data = data.with_columns(
    pl.col("Event").fill_null("")
)
list_indices = []
list_time_events = []
for i, value in enumerate(data['Event']):
    if value != "":
        print(i, value, data['Time'][i])
        list_indices.append(i)
        list_time_events.append(data['Time'][i])
numeric_cols = [c for c in column_names if c not in ["Event", "Event text"]]
data = data.with_columns(
        [pl.col(c).cast(pl.Float64) for c in numeric_cols]
    )

seconds_to_keep_before_the_trial = 10
MVC_trial_duration = 5
Training_duration = 30
Perturbation_trial = 30
data_points_to_keep_before_the_trial = int(10 * sampling_frequency)

print(data.columns)
plt.plot(data['Time'], data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'])
for line in list_time_events:
    plt.axvline(x=line, c='red')
plt.show()

MVC_T_1 = data.slice(list_indices[0] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)
MVC_T_2 = data.slice(list_indices[1] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)
MVC_T_3 = data.slice(list_indices[2] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)
Training_T_1 = data.slice(list_indices[3] - data_points_to_keep_before_the_trial, 30 + data_points_to_keep_before_the_trial)
Training_T_2 = data.slice(list_indices[4] - data_points_to_keep_before_the_trial, list_indices[5] - list_indices[4] + data_points_to_keep_before_the_trial)
Perturbation_trial = data.slice(list_indices[6] - data_points_to_keep_before_the_trial, 30 + data_points_to_keep_before_the_trial)

trials = [MVC_T_1, MVC_T_2, MVC_T_3, Training_T_1, Training_T_2, Perturbation_trial]
trials_names = ['MVC_T_1', 'MVC_T_2', 'MVC_T_3', 'Training_T_1', 'Training_T_2', 'Perturbation_trial']

for i, data in enumerate(trials):
    print(trials_names[i])

    time = data['Time'].to_numpy()
    left_Rx1_TSI_perc = data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'].to_numpy()
    left_RX1_TSI_Fit_Factor = data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor'].to_numpy()
    left_Rx1_Tx1_O2Hb = data['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = data['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = data['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
    left_Rx2_Tx1_O2Hb = data['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
    left_Rx2_Tx2_O2Hb = data['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
    left_Rx2_Tx3_O2Hb = data['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
    left_Rx1_Tx1_HHb = data['[9322] Rx1 - Tx1  HHb'].to_numpy()
    left_Rx1_Tx2_HHb = data['[9322] Rx1 - Tx2  HHb'].to_numpy()
    left_Rx1_Tx3_HHb = data['[9322] Rx1 - Tx3  HHb'].to_numpy()
    left_Rx2_Tx1_HHb = data['[9322] Rx2 - Tx1  HHb'].to_numpy()
    left_Rx2_Tx2_HHb = data['[9322] Rx2 - Tx2  HHb'].to_numpy()
    left_Rx2_Tx3_HHb = data['[9322] Rx2 - Tx3  HHb'].to_numpy()

    right_Rx3_TSI_perc = data['[9323] Rx3 - Tx4,Tx5,Tx6  TSI%'].to_numpy()
    right_RX3_TSI_Fit_Factor = data['[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor'].to_numpy()
    right_Rx3_Tx4_O2Hb = data['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = data['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = data['[9323] Rx3 - Tx6  O2Hb'].to_numpy()
    right_Rx4_Tx4_O2Hb = data['[9323] Rx4 - Tx4  O2Hb'].to_numpy()
    right_Rx4_Tx5_O2Hb = data['[9323] Rx4 - Tx5  O2Hb'].to_numpy()
    right_Rx4_Tx6_O2Hb = data['[9323] Rx4 - Tx6  O2Hb'].to_numpy()
    right_Rx3_Tx4_HHb = data['[9323] Rx3 - Tx4  HHb'].to_numpy()
    right_Rx3_Tx5_HHb = data['[9323] Rx3 - Tx5  HHb'].to_numpy()
    right_Rx3_Tx6_HHb = data['[9323] Rx3 - Tx6  HHb'].to_numpy()
    right_Rx4_Tx4_HHb = data['[9323] Rx4 - Tx4  HHb'].to_numpy()
    right_Rx4_Tx5_HHb = data['[9323] Rx4 - Tx5  HHb'].to_numpy()
    right_Rx4_Tx6_HHb = data['[9323] Rx4 - Tx6  HHb'].to_numpy()

    plt.plot(time, left_RX1_TSI_Fit_Factor, label='left_RX1_TSI_Fit_Factor')
    plt.plot(time, right_RX3_TSI_Fit_Factor, label='right_RX3_TSI_Fit_Factor')
    plt.axhline(y=90, label='Threshold of accurate data', c='red')
    plt.legend()
    plt.show()
    plot=True
    evaluation_left_Rx1_Tx1_O2Hb, peak_height_left_Rx1_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx1_O2Hb, 100,'[9322] Rx1 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx2_O2Hb, peak_height_left_Rx1_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx2_O2Hb, 100,'[9322] Rx1 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx3_O2Hb, peak_height_left_Rx1_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx3_O2Hb, 100,'[9322] Rx1 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx1_O2Hb, peak_height_left_Rx2_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx1_O2Hb, 100,'[9322] Rx2 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx2_O2Hb, peak_height_left_Rx2_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx2_O2Hb, 100,'[9322] Rx2 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx3_O2Hb, peak_height_left_Rx2_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx3_O2Hb, 100,'[9322] Rx2 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx1_HHb, peak_height_left_Rx1_Tx1_HHb = lb.fNIRS_check_quality(left_Rx1_Tx1_HHb, 100,'[9322] Rx1 - Tx1  HHb', plot=plot)
    evaluation_left_Rx1_Tx2_HHb, peak_height_left_Rx1_Tx2_HHb = lb.fNIRS_check_quality(left_Rx1_Tx2_HHb, 100,'[9322] Rx1 - Tx2  HHb', plot=plot)
    evaluation_left_Rx1_Tx3_HHb, peak_height_left_Rx1_Tx3_HHb = lb.fNIRS_check_quality(left_Rx1_Tx3_HHb, 100,'[9322] Rx1 - Tx3  HHb', plot=plot)
    evaluation_left_Rx2_Tx1_HHb, peak_height_left_Rx2_Tx1_HHb = lb.fNIRS_check_quality(left_Rx2_Tx1_HHb, 100,'[9322] Rx2 - Tx1  HHb', plot=plot)
    evaluation_left_Rx2_Tx2_HHb, peak_height_left_Rx2_Tx2_HHb = lb.fNIRS_check_quality(left_Rx2_Tx2_HHb, 100,'[9322] Rx2 - Tx2  HHb', plot=plot)
    evaluation_left_Rx2_Tx3_HHb, peak_height_left_Rx2_Tx3_HHb = lb.fNIRS_check_quality(left_Rx2_Tx3_HHb, 100,'[9322] Rx2 - Tx3  HHb', plot=plot)

    evaluation_right_Rx3_Tx4_O2Hb, peak_height_right_Rx3_Tx4_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx4_O2Hb, 100, '[9323] Rx3 - Tx4  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx5_O2Hb, peak_height_right_Rx3_Tx5_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx5_O2Hb, 100, '[9323] Rx3 - Tx5  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx6_O2Hb, peak_height_right_Rx3_Tx6_O2Hb = lb.fNIRS_check_quality(right_Rx3_Tx6_O2Hb, 100, '[9323] Rx3 - Tx6  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx4_O2Hb, peak_height_right_Rx4_Tx4_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx4_O2Hb, 100, '[9323] Rx4 - Tx4  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx5_O2Hb, peak_height_right_Rx4_Tx5_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx5_O2Hb, 100, '[9323] Rx4 - Tx5  O2Hb', plot=plot)
    evaluation_right_Rx4_Tx6_O2Hb, peak_height_right_Rx4_Tx6_O2Hb = lb.fNIRS_check_quality(right_Rx4_Tx6_O2Hb, 100, '[9323] Rx4 - Tx6  O2Hb', plot=plot)
    evaluation_right_Rx3_Tx4_HHb, peak_height_right_Rx3_Tx4_HHb = lb.fNIRS_check_quality(right_Rx3_Tx4_HHb, 100, '[9323] Rx3 - Tx4  HHb', plot=plot)
    evaluation_right_Rx3_Tx5_HHb, peak_height_right_Rx3_Tx5_HHb = lb.fNIRS_check_quality(right_Rx3_Tx5_HHb, 100, '[9323] Rx3 - Tx5  HHb', plot=plot)
    evaluation_right_Rx3_Tx6_HHb, peak_height_right_Rx3_Tx6_HHb = lb.fNIRS_check_quality(right_Rx3_Tx6_HHb, 100, '[9323] Rx3 - Tx6  HHb', plot=plot)
    evaluation_right_Rx4_Tx4_HHb, peak_height_right_Rx4_Tx4_HHb = lb.fNIRS_check_quality(right_Rx4_Tx4_HHb, 100, '[9323] Rx4 - Tx4  HHb', plot=plot)
    evaluation_right_Rx4_Tx5_HHb, peak_height_right_Rx4_Tx5_HHb = lb.fNIRS_check_quality(right_Rx4_Tx5_HHb, 100, '[9323] Rx4 - Tx5  HHb', plot=plot)
    evaluation_right_Rx4_Tx6_HHb, peak_height_right_Rx4_Tx6_HHb = lb.fNIRS_check_quality(right_Rx4_Tx6_HHb, 100, '[9323] Rx4 - Tx6  HHb', plot=plot)

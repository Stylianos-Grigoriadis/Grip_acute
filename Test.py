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
print(data['Time'][-1])
plt.plot(data['Time'], data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'])
for line in list_time_events:
    plt.axvline(x=line)
plt.show()

MVC_T_1 = data.slice(list_indices[0] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)
MVC_T_2 = data.slice(list_indices[1] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)
MVC_T_3 = data.slice(list_indices[2] - data_points_to_keep_before_the_trial, 5 + data_points_to_keep_before_the_trial)

Training_T_1 = data.slice(list_indices[3] - data_points_to_keep_before_the_trial, 30 + data_points_to_keep_before_the_trial)
Training_T_2 = data.slice(list_indices[4] - data_points_to_keep_before_the_trial, list_indices[5] - list_indices[4] + data_points_to_keep_before_the_trial)

Perturbation_trial = data.slice(list_indices[6] - data_points_to_keep_before_the_trial, 30 + data_points_to_keep_before_the_trial)

import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#     # The Rx1 and Rx3 are the long distance receiver
#     # The 9322 is left side and the 9323 is right side

plt.rcParams['font.family'] = 'serif'        # e.g., 'serif', 'sans-serif', 'monospace'
plt.rcParams['font.size'] = 16
directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen\Sine_1'
os.chdir(directory)

parts = directory.split(os.sep)
ID = parts[-1]
print(ID)

grip_directory = directory + r'\Grip data'
os.chdir(grip_directory)
training_set_1 = pd.read_csv(r'Training_1.csv', skiprows=2)
training_set_2 = pd.read_csv(r'Training_2.csv', skiprows=2)
training_set_3 = pd.read_csv(r'Training_3.csv', skiprows=2)
training_set_4 = pd.read_csv(r'Training_4.csv', skiprows=2)
training_set_5 = pd.read_csv(r'Training_5.csv', skiprows=2)
training_set_6 = pd.read_csv(r'Training_6.csv', skiprows=2)
training_set_7 = pd.read_csv(r'Training_7.csv', skiprows=2)
training_set_8 = pd.read_csv(r'Training_8.csv', skiprows=2)
training_set_9 = pd.read_csv(r'Training_9.csv', skiprows=2)
training_set_10 = pd.read_csv(r'Training_10.csv', skiprows=2)



training_set_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_1)
training_set_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_2)
training_set_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_3)
training_set_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_4)
training_set_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_5)
training_set_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_6)
training_set_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_7)
training_set_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_8)
training_set_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_9)
training_set_10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_10)


list_set = [training_set_1,
            training_set_2,
            training_set_3,
            training_set_4,
            training_set_5,
            training_set_6,
            training_set_7,
            training_set_8,
            training_set_9,
            training_set_10
            ]



directory_hemoglobin = directory + r'\Brain data'
name = f'Artinis_S1'
data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_22_events_plot(directory_hemoglobin, name)
# print(data)
# print(fs)
# print(list_indices)
# print(list_time_events)
# print(pre_event_indices)
# print(derived_end_indices)
# print(final_event_indices)
# print(list_training_sets)
# Erase the line below when you have only the training sets
list_training_sets = list_training_sets[6:-6]

# time = np.linspace(0, 40, len(training_set_1["Performance"]))
# plt.plot(time, training_set_1["Performance"], label="Performance")
# plt.plot(time, training_set_1["Target"], label="Target")
# plt.legend()
# plt.show()



def plot_force_and_stacked_o2hb_one_plotly(time, signal_dict, force_data, training_start_sec=10, force_duration=30, force_time_col=None, performance_col="Performance", target_col="Target", padding_percent=10, gap_percent=30, center_method="first", title="Force and stacked O2Hb traces", line_width=1.5, show=True):
    def to_numpy_1d(x):
        if hasattr(x, "to_numpy"):
            return np.asarray(x.to_numpy()).ravel()
        return np.asarray(x).ravel()

    def process_one_signal(y):
        y = to_numpy_1d(y).astype(float)

        finite_mask = np.isfinite(y)

        if not np.any(finite_mask):
            raise ValueError("One signal contains no finite values.")

        if center_method == "first":
            first_value = y[finite_mask][0]
            y = y - first_value

        elif center_method == "mean":
            y = y - np.nanmean(y)

        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        y_range = y_max - y_min

        if y_range == 0:
            y_range = 1

        padding = y_range * padding_percent / 100

        return {
            "y": y,
            "local_min": y_min - padding,
            "local_max": y_max + padding,
            "local_range": y_range + 2 * padding
        }

    time = to_numpy_1d(time).astype(float)
    time = time - time[0]

    performance = to_numpy_1d(force_data[performance_col]).astype(float)
    target = to_numpy_1d(force_data[target_col]).astype(float)

    if len(performance) != len(target):
        raise ValueError("Performance and Target have different lengths.")

    if force_time_col is not None:
        force_time = to_numpy_1d(force_data[force_time_col]).astype(float)
        force_time = force_time - force_time[0]
    else:
        force_time = np.linspace(0, force_duration, len(performance), endpoint=False)

    force_time = force_time + training_start_sec

    fig = go.Figure()

    groups = []

    for name, y in signal_dict.items():
        y = to_numpy_1d(y).astype(float)

        if len(y) != len(time):
            raise ValueError(f"{name} has a different length than time.")

        processed = process_one_signal(y)

        groups.append({
            "name": name,
            "type": "single",
            "time": time,
            "signals": {
                name: processed["y"]
            },
            "local_min": processed["local_min"],
            "local_max": processed["local_max"],
            "local_range": processed["local_range"]
        })

    force_min = np.nanmin([np.nanmin(performance), np.nanmin(target)])
    force_max = np.nanmax([np.nanmax(performance), np.nanmax(target)])
    force_range = force_max - force_min

    if force_range == 0:
        force_range = 1

    force_padding = force_range * padding_percent / 100

    groups.append({
        "name": "Force tracking",
        "type": "force",
        "time": force_time,
        "signals": {
            "Performance": performance,
            "Target": target
        },
        "local_min": force_min - force_padding,
        "local_max": force_max + force_padding,
        "local_range": force_range + 2 * force_padding
    })

    current_bottom = 0
    final_top = 0

    x_start = 0
    x_end = max(time[-1], force_time[-1])
    x_range = x_end - x_start
    label_x_position = x_end + x_range * 0.02

    for group_index, group in enumerate(groups):
        local_min = group["local_min"]
        local_range = group["local_range"]
        vertical_shift = current_bottom - local_min

        for signal_name, y in group["signals"].items():
            y_offset = y + vertical_shift

            if group["type"] == "force" and signal_name == "Target":
                line_style = dict(width=2, dash="dash")
            elif group["type"] == "force" and signal_name == "Performance":
                line_style = dict(width=2)
            else:
                line_style = dict(width=line_width)

            fig.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=y_offset,
                    mode="lines",
                    name=signal_name if group["type"] == "force" else group["name"],
                    line=line_style,
                    showlegend=True if group["type"] == "force" else False
                )
            )

        fig.add_annotation(
            x=label_x_position,
            y=current_bottom + local_range / 2,
            text=group["name"],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12)
        )

        final_top = current_bottom + local_range
        gap = local_range * gap_percent / 100
        current_bottom = final_top + gap

    fig.add_vrect(
        x0=0,
        x1=training_start_sec,
        fillcolor="lightgray",
        opacity=0.20,
        line_width=0
    )

    fig.add_vline(
        x=training_start_sec,
        line_width=2,
        line_dash="dash"
    )

    fig.add_annotation(
        x=training_start_sec,
        y=final_top,
        text="Training starts",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=13)
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showgrid=False,
        title_text="Stacked signals"
    )

    fig.update_xaxes(
        title_text="Time (s)",
        range=[x_start, x_end + x_range * 0.18]
    )

    fig.update_layout(
        title=title,
        height=max(700, 320 + len(groups) * 100),
        width=1400,
        margin=dict(l=80, r=260, t=90, b=70),
        template="plotly_white",
        hovermode="x unified"
    )

    if show:
        fig.show()

    return fig


for i in range(len(list_training_sets)):
    force_data = list_set[i]
    brain_data = list_training_sets[i]
    print(f'Set is {i + 1}')

    time = brain_data['Time'].to_numpy()
    time = time - time[0]
    print(time[-1])

    left_Rx1_TSI_perc = brain_data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI%'].to_numpy()
    left_RX1_TSI_Fit_Factor = brain_data['[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor'].to_numpy()
    left_Rx1_Tx1_O2Hb = brain_data['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = brain_data['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = brain_data['[9322] Rx1 - Tx3  O2Hb'].to_numpy()

    right_Rx3_TSI_perc = brain_data['[9323] Rx3 - Tx4,Tx5,Tx6  TSI%'].to_numpy()
    right_RX3_TSI_Fit_Factor = brain_data['[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor'].to_numpy()
    right_Rx3_Tx4_O2Hb = brain_data['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = brain_data['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = brain_data['[9323] Rx3 - Tx6  O2Hb'].to_numpy()

    left_Rx1_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)

    signal_dict = {
        "Left Rx1-Tx1 O2Hb": left_Rx1_Tx1_O2Hb,
        "Left Rx1-Tx2 O2Hb": left_Rx1_Tx2_O2Hb,
        "Left Rx1-Tx3 O2Hb": left_Rx1_Tx3_O2Hb,
        "Right Rx3-Tx4 O2Hb": right_Rx3_Tx4_O2Hb,
        "Right Rx3-Tx5 O2Hb": right_Rx3_Tx5_O2Hb,
        "Right Rx3-Tx6 O2Hb": right_Rx3_Tx6_O2Hb
    }

    plot_force_and_stacked_o2hb_one_plotly(
        time=time,
        signal_dict=signal_dict,
        force_data=force_data,
        training_start_sec=10,
        force_duration=30,
        padding_percent=10,
        gap_percent=1,
        center_method="first",
        title=f"Set {i + 1}: Force tracking and O2Hb response"
    )


for i in range(len(list_training_sets)):
    training_set = list_training_sets[i]
    # I will check it out but hypothetically the Rx1 and Rx3 are the long distance receiver
    # I will check it out but hypothetically the 9322 is left side and the 9323 is right side
    # print(training_set.columns)
    print(f'Set is {i+1}')

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

    plot_time = time - time[0]
    plt.plot(plot_time, left_RX1_TSI_Fit_Factor, label='left TSI Fit Factor', lw=3)
    plt.plot(plot_time, right_RX3_TSI_Fit_Factor, label='right TSI Fit Factor', lw=3)
    plt.axhline(y=90, label='Threshold of accurate data', c='red', lw=3)
    plt.legend()
    plt.show()

    if i+1 == 8:
        plot = False
    else:
        plot = False
    evaluation_left_Rx1_Tx1_O2Hb, peak_height_left_Rx1_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx1_O2Hb, 100, '[9322] Rx1 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx2_O2Hb, peak_height_left_Rx1_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx2_O2Hb, 100, '[9322] Rx1 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx3_O2Hb, peak_height_left_Rx1_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx1_Tx3_O2Hb, 100, '[9322] Rx1 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx1_O2Hb, peak_height_left_Rx2_Tx1_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx1_O2Hb, 100, '[9322] Rx2 - Tx1  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx2_O2Hb, peak_height_left_Rx2_Tx2_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx2_O2Hb, 100, '[9322] Rx2 - Tx2  O2Hb', plot=plot)
    evaluation_left_Rx2_Tx3_O2Hb, peak_height_left_Rx2_Tx3_O2Hb = lb.fNIRS_check_quality(left_Rx2_Tx3_O2Hb, 100, '[9322] Rx2 - Tx3  O2Hb', plot=plot)
    evaluation_left_Rx1_Tx1_HHb, peak_height_left_Rx1_Tx1_HHb = lb.fNIRS_check_quality(left_Rx1_Tx1_HHb, 100, '[9322] Rx1 - Tx1  HHb', plot=plot)
    evaluation_left_Rx1_Tx2_HHb, peak_height_left_Rx1_Tx2_HHb = lb.fNIRS_check_quality(left_Rx1_Tx2_HHb, 100, '[9322] Rx1 - Tx2  HHb', plot=plot)
    evaluation_left_Rx1_Tx3_HHb, peak_height_left_Rx1_Tx3_HHb = lb.fNIRS_check_quality(left_Rx1_Tx3_HHb, 100, '[9322] Rx1 - Tx3  HHb', plot=plot)
    evaluation_left_Rx2_Tx1_HHb, peak_height_left_Rx2_Tx1_HHb = lb.fNIRS_check_quality(left_Rx2_Tx1_HHb, 100, '[9322] Rx2 - Tx1  HHb', plot=plot)
    evaluation_left_Rx2_Tx2_HHb, peak_height_left_Rx2_Tx2_HHb = lb.fNIRS_check_quality(left_Rx2_Tx2_HHb, 100, '[9322] Rx2 - Tx2  HHb', plot=plot)
    evaluation_left_Rx2_Tx3_HHb, peak_height_left_Rx2_Tx3_HHb = lb.fNIRS_check_quality(left_Rx2_Tx3_HHb, 100, '[9322] Rx2 - Tx3  HHb', plot=plot)

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

    list_evaluation_left = [evaluation_left_Rx1_Tx1_O2Hb, evaluation_left_Rx1_Tx2_O2Hb, evaluation_left_Rx1_Tx3_O2Hb, evaluation_left_Rx2_Tx1_O2Hb, evaluation_left_Rx2_Tx2_O2Hb, evaluation_left_Rx2_Tx3_O2Hb, evaluation_left_Rx1_Tx1_HHb, evaluation_left_Rx1_Tx2_HHb, evaluation_left_Rx1_Tx3_HHb, evaluation_left_Rx2_Tx1_HHb, evaluation_left_Rx2_Tx2_HHb, evaluation_left_Rx2_Tx3_HHb]
    list_peak_height_left = [peak_height_left_Rx1_Tx1_O2Hb, peak_height_left_Rx1_Tx2_O2Hb, peak_height_left_Rx1_Tx3_O2Hb, peak_height_left_Rx2_Tx1_O2Hb, peak_height_left_Rx2_Tx2_O2Hb, peak_height_left_Rx2_Tx3_O2Hb, peak_height_left_Rx1_Tx1_HHb, peak_height_left_Rx1_Tx2_HHb, peak_height_left_Rx1_Tx3_HHb, peak_height_left_Rx2_Tx1_HHb, peak_height_left_Rx2_Tx2_HHb, peak_height_left_Rx2_Tx3_HHb]
    list_evaluation_right = [evaluation_right_Rx3_Tx4_O2Hb, evaluation_right_Rx3_Tx5_O2Hb, evaluation_right_Rx3_Tx6_O2Hb, evaluation_right_Rx4_Tx4_O2Hb, evaluation_right_Rx4_Tx5_O2Hb, evaluation_right_Rx4_Tx6_O2Hb, evaluation_right_Rx3_Tx4_HHb, evaluation_right_Rx3_Tx5_HHb, evaluation_right_Rx3_Tx6_HHb, evaluation_right_Rx4_Tx4_HHb, evaluation_right_Rx4_Tx5_HHb, evaluation_right_Rx4_Tx6_HHb]
    list_peak_height_right = [peak_height_right_Rx3_Tx4_O2Hb, peak_height_right_Rx3_Tx5_O2Hb, peak_height_right_Rx3_Tx6_O2Hb, peak_height_right_Rx4_Tx4_O2Hb, peak_height_right_Rx4_Tx5_O2Hb, peak_height_right_Rx4_Tx6_O2Hb, peak_height_right_Rx3_Tx4_HHb, peak_height_right_Rx3_Tx5_HHb, peak_height_right_Rx3_Tx6_HHb, peak_height_right_Rx4_Tx4_HHb, peak_height_right_Rx4_Tx5_HHb, peak_height_right_Rx4_Tx6_HHb]

    plt.plot(list_peak_height_left, label='left')
    plt.plot(list_peak_height_right, label='right')
    plt.axhline(y=12, label='Threshold of accurate cardiac rhythm', c='red')
    plt.legend()
    plt.show()



    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx1_O2Hb, fs=fs, thresh_z=4, plot=True)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx2_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx3_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx1_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx2_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx3_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx1_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx2_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx1_Tx3_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx1_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=left_Rx2_Tx2_HHb, fs=fs, thresh_z=4, plot=plot)

    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx4_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx5_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx6_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx4_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx5_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx6_O2Hb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx4_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx5_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx3_Tx6_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx4_HHb, fs=fs, thresh_z=4, plot=plot)
    lb.detect_motion_mask_from_movstd(time_window=2, signal=right_Rx4_Tx5_HHb, fs=fs, thresh_z=4, plot=plot)




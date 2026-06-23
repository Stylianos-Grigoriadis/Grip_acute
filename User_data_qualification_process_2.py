import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import plotly.graph_objects as go
import plotly.io as pio
import json

from pathlib import Path
from datetime import datetime
from html import escape
from plotly.subplots import make_subplots

# =========================
# SETTINGS
# =========================
directory = Path(
    r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen\White_2'
)

ID = directory.name
artinis_file_name = "Artinis_" + ID[0] + ID.split("_")[1]
# artinis_file_name = 'Artinis_W1'

grip_directory = directory / 'Grip data'

participants_excel_path = Path(
    r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals\Participants.xlsx'
)

number_of_training_sets = 10
save_directory = Path(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Figures\Reports')
report_path = save_directory / f'{ID}_prequalification_report.html'

force_color = 'blue'
target_color = 'red'
spatial_error_color = 'black'
spatial_error_band_color = 'rgba(0, 0, 0, 0.18)'


# =========================
# PERTURBATION SETTINGS
# =========================
sampling_frequency = 100
low_pass_filter_frequency = 15
sd_factor = 3
time_window = 1
time_threshold = 3

perturbation_spatial_error_color = 'black'
perturbation_threshold_color = 'black'
perturbation_adaptation_color = 'red'
perturbation_instance_color = 'gray'
perturbation_window_color = 'rgba(128, 128, 128, 0.25)'


# =========================
# HEMOGLOBIN SETTINGS
# =========================
brain_directory = directory / 'Brain data'

hemoglobin_training_start_sec = 10
hemoglobin_force_duration = 30

hemoglobin_low_frequency = 0.01
hemoglobin_high_frequency = 0.30
hemoglobin_filter_order = 4

hemoglobin_padding_percent = 10
hemoglobin_gap_percent = 1
hemoglobin_center_method = 'first'


# =========================
# FUNCTIONS
# =========================
def load_participant_information(participants_excel_path, ID):
    """
    Loads participant information from the Participants.xlsx file.
    """
    information = pd.read_excel(participants_excel_path)

    matching_row = information.loc[information["ID"] == ID]

    if matching_row.empty:
        raise ValueError(f"No participant with ID '{ID}' was found in: {participants_excel_path}")

    date_of_collection = matching_row.iloc[0, 2]

    if hasattr(date_of_collection, "date"):
        date_of_collection = date_of_collection.date()

    age = matching_row.iloc[0, 3]
    weight = matching_row.iloc[0, 4]
    height = matching_row.iloc[0, 5]
    MVC = matching_row.iloc[0, 6]
    dominant_hand = matching_row.iloc[0, 7]

    participant_information = {
        "ID": str(ID),
        "date_of_collection": str(date_of_collection),
        "age": str(round(age)) + " years old",
        "weight": str(round(weight)) + " kg",
        "height": str(round(height)) + " cm",
        "MVC": str(round(MVC * 9.81)) + "N",
        "dominant_hand": str(dominant_hand)
    }

    return participant_information


def create_participant_information_html(participant_information):
    """
    Creates the HTML text for the Basic Information section.
    """
    html = f"""
        <pre style="font-family: Georgia, serif; font-size: 18px; line-height: 1.8;">
ID:                 {escape(participant_information["ID"])}
Date of collection: {escape(participant_information["date_of_collection"])}
Age:                {escape(participant_information["age"])}
Weight:             {escape(participant_information["weight"])}
Height:             {escape(participant_information["height"])}
MVC:                {escape(participant_information["MVC"])}
Dominant arm:       {escape(participant_information["dominant_hand"])}
        </pre>
    """

    return html


def create_perturbation_force_target_data_for_javascript(upward_perturbation_results, downward_perturbation_results):
    """
    Prepares Performance and Target data so JavaScript can draw a popup graph
    when a perturbation subplot is clicked.
    """
    perturbation_force_target_data = {}

    all_results = upward_perturbation_results + downward_perturbation_results

    for result in all_results:
        df = result['df']
        perturbation_index = result['perturbation_index']
        perturbation_time = df['Time'].iloc[perturbation_index]

        plot_key = result['plot_key']

        force_time_shifted = df['Time'] - perturbation_time
        target_time_shifted = df['ClosestSampleTime'] - perturbation_time

        perturbation_force_target_data[plot_key] = {
            'Name': result['name'],
            'Direction': result['direction'],
            'Phase': result['phase'],
            'Trial_Number': result['trial_number'],
            'Force_Time': clean_numeric_list(force_time_shifted),
            'Performance': clean_numeric_list(df['Performance']),
            'Target_Time': clean_numeric_list(target_time_shifted),
            'Target': clean_numeric_list(df['Target'])
        }

    return perturbation_force_target_data


def read_and_preprocess_perturbation_file(file_path, sampling_frequency, low_pass_filter_frequency):
    """
    Reads one perturbation or isometric file, synchronizes it,
    and low-pass filters the Performance column.
    """
    df = pd.read_csv(file_path, skiprows=2)
    df = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    df['Performance'] = lib.Butterworth(
        sampling_frequency,
        low_pass_filter_frequency,
        df['Performance'].to_numpy()
    )

    return df


def load_perturbation_trials(grip_directory, sampling_frequency, low_pass_filter_frequency):
    """
    Loads all perturbation trials and the isometric trials.
    """
    upward_file_info = [
        ('Pre', 1, 'Pre_Pert_up_1.csv'),
        ('Pre', 2, 'Pre_Pert_up_2.csv'),
        ('Pre', 3, 'Pre_Pert_up_3.csv'),
        ('Post', 1, 'Post_Pert_up_1.csv'),
        ('Post', 2, 'Post_Pert_up_2.csv'),
        ('Post', 3, 'Post_Pert_up_3.csv')
    ]

    downward_file_info = [
        ('Pre', 1, 'Pre_Pert_down_1.csv'),
        ('Pre', 2, 'Pre_Pert_down_2.csv'),
        ('Pre', 3, 'Pre_Pert_down_3.csv'),
        ('Post', 1, 'Post_Pert_down_1.csv'),
        ('Post', 2, 'Post_Pert_down_2.csv'),
        ('Post', 3, 'Post_Pert_down_3.csv')
    ]

    upward_trials = []
    downward_trials = []

    for phase, trial_number, file_name in upward_file_info:
        file_path = grip_directory / file_name

        if not file_path.exists():
            raise FileNotFoundError(f'This file was not found: {file_path}')

        df = read_and_preprocess_perturbation_file(
            file_path=file_path,
            sampling_frequency=sampling_frequency,
            low_pass_filter_frequency=low_pass_filter_frequency
        )

        upward_trials.append({
            'phase': phase,
            'trial_number': trial_number,
            'name': f'{phase} Up {trial_number}',
            'file_name': file_name,
            'df': df
        })

    for phase, trial_number, file_name in downward_file_info:
        file_path = grip_directory / file_name

        if not file_path.exists():
            raise FileNotFoundError(f'This file was not found: {file_path}')

        df = read_and_preprocess_perturbation_file(
            file_path=file_path,
            sampling_frequency=sampling_frequency,
            low_pass_filter_frequency=low_pass_filter_frequency
        )

        downward_trials.append({
            'phase': phase,
            'trial_number': trial_number,
            'name': f'{phase} Down {trial_number}',
            'file_name': file_name,
            'df': df
        })

    isometric_high_path = grip_directory / 'Isometric_high.csv'
    isometric_low_path = grip_directory / 'Isometric_low.csv'

    if not isometric_high_path.exists():
        raise FileNotFoundError(f'This file was not found: {isometric_high_path}')

    if not isometric_low_path.exists():
        raise FileNotFoundError(f'This file was not found: {isometric_low_path}')

    isometric_high = read_and_preprocess_perturbation_file(
        file_path=isometric_high_path,
        sampling_frequency=sampling_frequency,
        low_pass_filter_frequency=low_pass_filter_frequency
    )

    isometric_low = read_and_preprocess_perturbation_file(
        file_path=isometric_low_path,
        sampling_frequency=sampling_frequency,
        low_pass_filter_frequency=low_pass_filter_frequency
    )

    return upward_trials, downward_trials, isometric_high, isometric_low


def calculate_isometric_thresholds(isometric_high, isometric_low, time_threshold):
    """
    Calculates the mean and SD of spatial error from the isometric trials.
    """
    isometric_high = isometric_high[isometric_high['Time'] > time_threshold].reset_index(drop=True).copy()
    isometric_low = isometric_low[isometric_low['Time'] > time_threshold].reset_index(drop=True).copy()

    spatial_errors_high = lb.spatial_error(isometric_high)
    spatial_errors_low = lb.spatial_error(isometric_low)

    mean_spatial_error_high = np.mean(spatial_errors_high)
    mean_spatial_error_low = np.mean(spatial_errors_low)

    sd_spatial_error_high = np.std(spatial_errors_high)
    sd_spatial_error_low = np.std(spatial_errors_low)

    return {
        'mean_high': mean_spatial_error_high,
        'sd_high': sd_spatial_error_high,
        'mean_low': mean_spatial_error_low,
        'sd_low': sd_spatial_error_low
    }


def analyze_perturbation_trial(df, sd_factor, time_window, name, mean_spatial_error_isometric_trials, sd_spatial_error_isometric_trials):
    """
    Recreates the SD adaptation logic and returns everything needed for Plotly plotting.
    """
    df = df.copy()
    df = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    target_change_indices = df[df['Target'] != df['Target'].shift(1)].index.tolist()

    if len(target_change_indices) < 2:
        perturbation_index = target_change_indices[0]
    else:
        perturbation_index = target_change_indices[1]

    spatial_er = lb.spatial_error(df)

    mean_isometric_trial = mean_spatial_error_isometric_trials
    sd_isometric_trial = sd_spatial_error_isometric_trials
    threshold = mean_isometric_trial + (sd_isometric_trial * sd_factor)

    time_shifted = (df['Time'] - df['Time'].iloc[perturbation_index]).to_numpy()

    time_of_adaptation_sd = None
    time_until_spatial_error_is_lower_than_threshold = None

    target_time = df['Time'].iloc[-1] - time_window
    idx = (df['Time'] - target_time).abs().idxmin()
    pos_idx = df.index.get_loc(idx)
    last_pos = len(df) - 1
    last_index_of_iteration = last_pos - pos_idx

    for i in range(len(spatial_er) - last_index_of_iteration):
        if i >= perturbation_index:
            start_time_window_position = i
            end_time_window = df['Time'].iloc[start_time_window_position] + time_window

            if end_time_window > df['Time'].iloc[-1]:
                break

            end_idx_label = (df['Time'] - end_time_window).abs().idxmin()
            end_time_window_position = df.index.get_loc(end_idx_label)

            consecutive_values = end_time_window_position - start_time_window_position

            if consecutive_values <= 0:
                continue

            consecutive_values_list = np.arange(0, consecutive_values, 1)

            if all(
                spatial_er[i + j] < threshold
                for j in consecutive_values_list
            ):
                time_of_adaptation_sd = df['Time'].iloc[i] - df['Time'].iloc[perturbation_index]
                time_until_spatial_error_is_lower_than_threshold = (
                    df['Time'].iloc[end_time_window_position] - df['Time'].iloc[perturbation_index]
                )
                break

    return {
        'name': name,
        'df': df,
        'spatial_error': spatial_er,
        'time_shifted': time_shifted,
        'perturbation_index': perturbation_index,
        'threshold': threshold,
        'mean_isometric_trial': mean_isometric_trial,
        'sd_isometric_trial': sd_isometric_trial,
        'time_of_adaptation_sd': time_of_adaptation_sd,
        'time_until_spatial_error_is_lower_than_threshold': time_until_spatial_error_is_lower_than_threshold
    }


def analyze_perturbation_group(trials, sd_factor, time_window, mean_spatial_error_isometric_trials, sd_spatial_error_isometric_trials, direction):
    """
    Applies the perturbation analysis to all trials of one group.
    """
    analyzed_trials = []

    for trial in trials:
        result = analyze_perturbation_trial(
            df=trial['df'],
            sd_factor=sd_factor,
            time_window=time_window,
            name=trial['name'],
            mean_spatial_error_isometric_trials=mean_spatial_error_isometric_trials,
            sd_spatial_error_isometric_trials=sd_spatial_error_isometric_trials
        )

        result['phase'] = trial['phase']
        result['trial_number'] = trial['trial_number']
        result['direction'] = direction
        result['plot_key'] = f"{direction}_{trial['phase']}_{trial['trial_number']}"

        analyzed_trials.append(result)

    analyzed_trials = sorted(
        analyzed_trials,
        key=lambda x: (0 if x['phase'] == 'Pre' else 1, x['trial_number'])
    )

    return analyzed_trials


def create_perturbation_figure(analyzed_trials, figure_title):
    """
    Creates a 2 x 3 Plotly figure:
    Row 1 = Pre
    Row 2 = Post
    Columns = trials 1, 2, 3
    """
    subplot_titles = []

    for result in analyzed_trials:
        if result['time_of_adaptation_sd'] is None:
            subtitle = f"{result['phase']} {result['trial_number']}<br>No adaptation"
        else:
            subtitle = f"{result['phase']} {result['trial_number']}<br>t = {result['time_of_adaptation_sd']:.2f} s"

        subplot_titles.append(subtitle)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    for i, result in enumerate(analyzed_trials):
        row = 1 if result['phase'] == 'Pre' else 2
        col = result['trial_number']

        x = result['time_shifted']
        y = result['spatial_error']
        threshold = result['threshold']
        adaptation_time = result['time_of_adaptation_sd']
        check_window_end = result['time_until_spatial_error_is_lower_than_threshold']
        plot_key = result['plot_key']

        y_max = np.nanmax([np.nanmax(y), threshold]) * 1.10
        if y_max <= 0:
            y_max = 1

        show_legend = True if i == 0 else False

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Spatial Error',
                legendgroup='Spatial Error',
                showlegend=show_legend,
                line=dict(color=perturbation_spatial_error_color),
                customdata=[plot_key] * len(x),
                hovertemplate='Time: %{x}<br>Spatial Error: %{y}<extra></extra>'
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[threshold, threshold],
                mode='lines',
                name='Threshold',
                legendgroup='Threshold',
                showlegend=show_legend,
                line=dict(color=perturbation_threshold_color, dash='dot', width=3),
                hovertemplate='Threshold: %{y}<extra></extra>'
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, y_max],
                mode='lines',
                name='Perturbation instance',
                legendgroup='Perturbation instance',
                showlegend=show_legend,
                line=dict(color=perturbation_instance_color, dash='dash', width=3),
                hoverinfo='skip'
            ),
            row=row,
            col=col
        )

        if adaptation_time is not None:
            fig.add_trace(
                go.Scatter(
                    x=[adaptation_time, adaptation_time],
                    y=[0, y_max],
                    mode='lines',
                    name='Adaptation instance',
                    legendgroup='Adaptation instance',
                    showlegend=show_legend,
                    line=dict(color=perturbation_adaptation_color, width=3),
                    hoverinfo='skip'
                ),
                row=row,
                col=col
            )

            if check_window_end is not None:
                fig.add_vrect(
                    x0=adaptation_time,
                    x1=check_window_end,
                    fillcolor='gray',
                    opacity=0.25,
                    line_width=0,
                    row=row,
                    col=col
                )

        fig.update_xaxes(title_text='Time (sec)', row=row, col=col)
        fig.update_yaxes(title_text='Force difference (kg)', row=row, col=col)

    fig.update_layout(
        title=dict(
            text=figure_title,
            x=0.5,
            xanchor='center'
        ),
        height=900,
        width=1300,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=160, l=70, r=40, b=70)
    )

    fig.add_annotation(
        text='Pre',
        xref='paper',
        yref='paper',
        x=-0.06,
        y=0.80,
        showarrow=False,
        textangle=-90,
        font=dict(size=16)
    )

    fig.add_annotation(
        text='Post',
        xref='paper',
        yref='paper',
        x=-0.06,
        y=0.22,
        showarrow=False,
        textangle=-90,
        font=dict(size=16)
    )

    return fig


def read_training_file(file_path):
    """
    Reads and synchronizes one training file.
    """
    df = pd.read_csv(file_path, skiprows=2)
    df = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df)
    return df


def load_all_training_sets(grip_directory, number_of_training_sets):
    """
    Loads all training sets from Training_1.csv to Training_10.csv.
    """
    training_sets = []

    for i in range(1, number_of_training_sets + 1):
        file_path = grip_directory / f'Training_{i}.csv'

        if not file_path.exists():
            raise FileNotFoundError(f'This file was not found: {file_path}')

        df = read_training_file(file_path)
        training_sets.append(df)

    return training_sets


def calculate_spatial_error_for_one_set(df):
    """
    Calculates the absolute spatial error between force output and target.
    """
    df = df.copy()

    Performance = df['Performance'].to_numpy()
    Target = df['Target'].to_numpy()

    Spatial_Error = np.abs(Performance - Target)

    df['Spatial_Error'] = Spatial_Error

    return df


def calculate_spatial_error_for_all_sets(training_sets):
    """
    Applies the spatial error calculation to all training sets.
    """
    spatial_error_sets = []

    for df in training_sets:
        df_with_error = calculate_spatial_error_for_one_set(df)
        spatial_error_sets.append(df_with_error)

    return spatial_error_sets


def calculate_spatial_error_summary(spatial_error_sets):
    """
    Calculates mean spatial error and SD for each training set.
    """
    summary_rows = []

    for i, df in enumerate(spatial_error_sets, start=1):
        spatial_error = df['Spatial_Error'].to_numpy()

        mean_error = np.nanmean(spatial_error)
        sd_error = np.nanstd(spatial_error, ddof=1)

        summary_rows.append({
            'Set': f'Set {i}',
            'Mean_Spatial_Error': mean_error,
            'SD_Spatial_Error': sd_error,
            'Upper_Bound': mean_error + sd_error,
            'Lower_Bound': mean_error - sd_error
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


def clean_numeric_list(series):
    """
    Converts a pandas Series to a JSON-safe list.
    NaN values become None, which becomes null in JavaScript.
    """
    clean_list = []

    for value in series:
        if pd.isna(value):
            clean_list.append(None)
        else:
            clean_list.append(float(value))

    return clean_list


def create_spatial_error_data_for_javascript(spatial_error_sets):
    """
    Prepares the spatial error data so JavaScript can draw the popup graph.
    """
    spatial_error_data = {}

    for i, df in enumerate(spatial_error_sets, start=1):
        spatial_error_data[str(i)] = {
            'Time': clean_numeric_list(df['Time']),
            'Spatial_Error': clean_numeric_list(df['Spatial_Error'])
        }

    return spatial_error_data


def add_plotly_figure_to_report(html_parts, title, fig, div_id, section_id, include_plotlyjs=True, page_class='figure-section'):
    """
    Adds one interactive Plotly figure to the HTML report.
    """
    fig_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        div_id=div_id,
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'responsive': True
        }
    )

    html_parts.append(f"""
    <section id="{section_id}" class="{page_class}">
        <h2>{escape(title)}</h2>
        {fig_html}
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
    """)


def create_training_overview_figure(training_sets):
    """
    Creates one interactive 5 x 2 Plotly overview figure for all training sets.
    """
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=[f'Set {i}' for i in range(1, len(training_sets) + 1)],
        vertical_spacing=0.07,
        horizontal_spacing=0.08
    )

    for i, df in enumerate(training_sets):
        set_number = i + 1
        row = (i // 2) + 1
        col = (i % 2) + 1

        show_legend = True if i == 0 else False

        fig.add_trace(
            go.Scatter(
                x=df['Time'],
                y=df['Performance'],
                mode='lines',
                name='Force Output',
                legendgroup='Force Output',
                showlegend=show_legend,
                line=dict(color=force_color),
                customdata=[set_number] * len(df),
                hovertemplate=(
                    'Set: %{customdata}<br>'
                    'Time: %{x}<br>'
                    'Force Output: %{y}<extra></extra>'
                )
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=df['ClosestSampleTime'],
                y=df['Target'],
                mode='lines',
                name='Target',
                legendgroup='Target',
                showlegend=show_legend,
                line=dict(color=target_color),
                customdata=[set_number] * len(df),
                hovertemplate=(
                    'Set: %{customdata}<br>'
                    'Time: %{x}<br>'
                    'Target: %{y}<extra></extra>'
                )
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text='Time', row=row, col=col)
        fig.update_yaxes(title_text='Force', row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f'{ID} - Training Sets Overview',
            x=0.5,
            xanchor='center'
        ),
        height=1600,
        width=1200,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=140, l=60, r=40, b=60)
    )

    return fig


def create_spatial_error_summary_figure(spatial_error_summary_df):
    """
    Creates a full spatial error summary graph across all training sets.
    The line is the mean spatial error.
    The shaded area is mean ± SD.
    """
    x_values = spatial_error_summary_df['Set']
    mean_values = spatial_error_summary_df['Mean_Spatial_Error']
    upper_values = spatial_error_summary_df['Upper_Bound']
    lower_values = spatial_error_summary_df['Lower_Bound']

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=upper_values,
            mode='lines',
            name='Mean + SD',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=lower_values,
            mode='lines',
            name='Mean ± SD',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=spatial_error_band_color,
            hovertemplate=(
                'Set: %{x}<br>'
                'Lower Bound: %{y}<extra></extra>'
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=mean_values,
            mode='lines+markers',
            name='Mean Spatial Error',
            line=dict(color=spatial_error_color, width=3),
            marker=dict(size=8),
            hovertemplate=(
                'Set: %{x}<br>'
                'Mean Spatial Error: %{y}<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=f'{ID} - Spatial Error Summary',
            x=0.5,
            xanchor='center'
        ),
        height=650,
        width=1200,
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(
            title='Training Set'
        ),
        yaxis=dict(
            title='Spatial Error',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=130, l=70, r=40, b=70)
    )

    return fig


def create_popup_javascript(spatial_error_data):
    """
    Creates the JavaScript that opens the spatial error graph when a training set is clicked.
    """
    spatial_error_json = json.dumps(spatial_error_data)

    javascript = f"""
    <script>
        const spatialErrorData = {spatial_error_json};

        const trainingPlot = document.getElementById('training_overview_plot');
        const modal = document.getElementById('spatial-error-modal');
        const closeButton = document.getElementById('close-spatial-error-modal');

        function showSpatialErrorGraph(setNumber) {{
            const setKey = String(setNumber);
            const setData = spatialErrorData[setKey];

            if (!setData) {{
                alert('No spatial error data found for Set ' + setNumber);
                return;
            }}

            document.getElementById('spatial-error-title').innerText = 'Spatial Error - Set ' + setNumber;

            const spatialErrorTrace = {{
                x: setData.Time,
                y: setData.Spatial_Error,
                mode: 'lines',
                type: 'scatter',
                name: 'Spatial Error',
                line: {{
                    color: '{spatial_error_color}'
                }},
                hovertemplate: 'Time: %{{x}}<br>Spatial Error: %{{y}}<extra></extra>'
            }};

            const layout = {{
                title: {{
                    text: 'Set ' + setNumber,
                    x: 0.5,
                    xanchor: 'center'
                }},
                height: 550,
                template: 'plotly_white',
                hovermode: 'x unified',
                xaxis: {{
                    title: 'Time'
                }},
                yaxis: {{
                    title: 'Spatial Error',
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: 'gray'
                }},
                margin: {{
                    t: 80,
                    l: 70,
                    r: 40,
                    b: 60
                }}
            }};

            const config = {{
                scrollZoom: true,
                displayModeBar: true,
                responsive: true
            }};

            Plotly.newPlot('spatial-error-plot', [spatialErrorTrace], layout, config);

            modal.style.display = 'block';
        }}

        if (trainingPlot) {{
            trainingPlot.on('plotly_click', function(clickData) {{
                if (!clickData.points || clickData.points.length === 0) {{
                    return;
                }}

                const clickedPoint = clickData.points[0];
                const setNumber = clickedPoint.customdata;

                showSpatialErrorGraph(setNumber);
            }});
        }}

        closeButton.onclick = function() {{
            modal.style.display = 'none';
        }};

        window.onclick = function(event) {{
            if (event.target === modal) {{
                modal.style.display = 'none';
            }}
        }};

        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                modal.style.display = 'none';
            }}
        }});
    </script>
    """

    return javascript


def create_perturbation_force_target_popup_javascript(perturbation_force_target_data):
    """
    Creates JavaScript that opens the hidden Performance + Target graph
    when a perturbation spatial-error subplot is clicked.
    """
    perturbation_force_target_json = json.dumps(perturbation_force_target_data)

    javascript = f"""
    <script>
        const perturbationForceTargetData = {perturbation_force_target_json};

        const upwardPerturbationPlot = document.getElementById('upward_perturbations_plot');
        const downwardPerturbationPlot = document.getElementById('downward_perturbations_plot');

        const perturbationForceTargetModal = document.getElementById('perturbation-force-target-modal');
        const closePerturbationForceTargetButton = document.getElementById('close-perturbation-force-target-modal');

        function showPerturbationForceTargetGraph(plotKey) {{
            const trialData = perturbationForceTargetData[plotKey];

            if (!trialData) {{
                alert('No Performance/Target data found for this perturbation.');
                return;
            }}

            document.getElementById('perturbation-force-target-title').innerText =
                trialData.Direction + ' Perturbation - ' + trialData.Phase + ' Trial ' + trialData.Trial_Number;

            const performanceTrace = {{
                x: trialData.Force_Time,
                y: trialData.Performance,
                mode: 'lines',
                type: 'scatter',
                name: 'Force Output',
                line: {{
                    color: '{force_color}'
                }},
                hovertemplate: 'Time: %{{x}}<br>Force Output: %{{y}}<extra></extra>'
            }};

            const targetTrace = {{
                x: trialData.Target_Time,
                y: trialData.Target,
                mode: 'lines',
                type: 'scatter',
                name: 'Target',
                line: {{
                    color: '{target_color}'
                }},
                hovertemplate: 'Time: %{{x}}<br>Target: %{{y}}<extra></extra>'
            }};

            const layout = {{
                title: {{
                    text: trialData.Name,
                    x: 0.5,
                    xanchor: 'center'
                }},
                height: 550,
                template: 'plotly_white',
                hovermode: 'x unified',
                xaxis: {{
                    title: 'Time from perturbation (sec)'
                }},
                yaxis: {{
                    title: 'Force'
                }},
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.03,
                    xanchor: 'center',
                    x: 0.5
                }},
                margin: {{
                    t: 100,
                    l: 70,
                    r: 40,
                    b: 60
                }}
            }};

            const config = {{
                scrollZoom: true,
                displayModeBar: true,
                responsive: true
            }};

            Plotly.newPlot(
                'perturbation-force-target-plot',
                [performanceTrace, targetTrace],
                layout,
                config
            );

            perturbationForceTargetModal.style.display = 'block';
        }}

        function connectPerturbationClick(plotElement) {{
            if (!plotElement) {{
                return;
            }}

            plotElement.on('plotly_click', function(clickData) {{
                if (!clickData.points || clickData.points.length === 0) {{
                    return;
                }}

                const clickedPoint = clickData.points[0];
                const plotKey = clickedPoint.customdata;

                if (!plotKey) {{
                    return;
                }}

                showPerturbationForceTargetGraph(plotKey);
            }});
        }}

        connectPerturbationClick(upwardPerturbationPlot);
        connectPerturbationClick(downwardPerturbationPlot);

        closePerturbationForceTargetButton.onclick = function() {{
            perturbationForceTargetModal.style.display = 'none';
        }};

        window.addEventListener('click', function(event) {{
            if (event.target === perturbationForceTargetModal) {{
                perturbationForceTargetModal.style.display = 'none';
            }}
        }});

        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                perturbationForceTargetModal.style.display = 'none';
            }}
        }});
    </script>
    """

    return javascript


def to_numpy_1d(x):
    """
    Converts pandas Series, lists, or arrays to a 1D numpy array.
    """
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy()).ravel()

    return np.asarray(x).ravel()


def process_one_stacked_signal(y, padding_percent, center_method):
    """
    Centers and prepares one signal for stacked plotting.
    """
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


def load_hemoglobin_training_sets(brain_directory, artinis_file_name):
    """
    Loads the Artinis file and keeps only the middle 10 events,
    which correspond to the 10 training sets.
    """
    data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_10_events_plot(
        brain_directory,
        artinis_file_name,
        write_manual_events_to_excel=False
    )


    if len(list_training_sets) != number_of_training_sets:
        raise ValueError(
            f"Expected {number_of_training_sets} hemoglobin training sets, "
            f"but found {len(list_training_sets)} after list_training_sets[6:-6]."
        )

    return data, fs, list_training_sets


def extract_filtered_o2hb_signals(brain_data, fs):
    """
    Extracts and band-pass filters the six O2Hb signals.
    """
    left_Rx1_Tx1_O2Hb = brain_data['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = brain_data['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = brain_data['[9322] Rx1 - Tx3  O2Hb'].to_numpy()

    right_Rx3_Tx4_O2Hb = brain_data['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = brain_data['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = brain_data['[9323] Rx3 - Tx6  O2Hb'].to_numpy()

    # left_Rx1_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)
    # left_Rx1_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)
    # left_Rx1_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)

    # right_Rx3_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)
    # right_Rx3_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)
    # right_Rx3_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_O2Hb, fs, low=hemoglobin_low_frequency, high=hemoglobin_high_frequency, order=hemoglobin_filter_order, plot=False, demean=False)

    signal_dict = {
        "Left Rx1-Tx1 O2Hb": left_Rx1_Tx1_O2Hb,
        "Left Rx1-Tx2 O2Hb": left_Rx1_Tx2_O2Hb,
        "Left Rx1-Tx3 O2Hb": left_Rx1_Tx3_O2Hb,
        "Right Rx3-Tx4 O2Hb": right_Rx3_Tx4_O2Hb,
        "Right Rx3-Tx5 O2Hb": right_Rx3_Tx5_O2Hb,
        "Right Rx3-Tx6 O2Hb": right_Rx3_Tx6_O2Hb
    }

    return signal_dict


def create_force_and_stacked_o2hb_figure(time, signal_dict, force_data, set_number, training_start_sec=10, force_duration=30, force_time_col=None, performance_col="Performance", target_col="Target", padding_percent=10, gap_percent=1, center_method="first", title="Force tracking and O2Hb response", line_width=1.5):
    """
    Creates one Plotly figure with stacked O2Hb signals and force tracking.
    """
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

    for signal_name, y in signal_dict.items():
        y = to_numpy_1d(y).astype(float)

        if len(y) != len(time):
            raise ValueError(f"{signal_name} has a different length than time.")

        processed = process_one_stacked_signal(
            y=y,
            padding_percent=padding_percent,
            center_method=center_method
        )

        groups.append({
            "name": signal_name,
            "type": "single",
            "time": time,
            "signals": {
                signal_name: processed["y"]
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

    for group in groups:
        local_min = group["local_min"]
        local_range = group["local_range"]
        vertical_shift = current_bottom - local_min

        for signal_name, y in group["signals"].items():
            y_offset = y + vertical_shift

            if group["type"] == "force" and signal_name == "Target":
                line_style = dict(color=target_color, width=2, dash="dash")
                trace_name = "Target"

            elif group["type"] == "force" and signal_name == "Performance":
                line_style = dict(color=force_color, width=2)
                trace_name = "Force Output"

            else:
                line_style = dict(width=line_width)
                trace_name = group["name"]

            fig.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=y_offset,
                    mode="lines",
                    name=trace_name,
                    line=line_style,
                    showlegend=True if group["type"] == "force" else False,
                    hovertemplate=(
                        "Time: %{x}<br>"
                        f"{trace_name}: " + "%{y}<extra></extra>"
                    )
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
        title=dict(
            text=title,
            x=0.5,
            xanchor="center"
        ),
        height=max(700, 320 + len(groups) * 100),
        width=1400,
        margin=dict(l=80, r=260, t=90, b=70),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_all_hemoglobin_training_figures(training_sets, hemoglobin_training_sets, fs):
    """
    Creates one stacked O2Hb + force tracking figure for each training set.
    """
    hemoglobin_figures = []

    for i in range(len(hemoglobin_training_sets)):
        force_data = training_sets[i]
        brain_data = hemoglobin_training_sets[i]

        time = brain_data['Time'].to_numpy()
        time = time - time[0]

        signal_dict = extract_filtered_o2hb_signals(
            brain_data=brain_data,
            fs=fs
        )

        fig = create_force_and_stacked_o2hb_figure(
            time=time,
            signal_dict=signal_dict,
            force_data=force_data,
            set_number=i + 1,
            training_start_sec=hemoglobin_training_start_sec,
            force_duration=hemoglobin_force_duration,
            padding_percent=hemoglobin_padding_percent,
            gap_percent=hemoglobin_gap_percent,
            center_method=hemoglobin_center_method,
            title=f"Set {i + 1}: Force tracking and O2Hb response"
        )

        hemoglobin_figures.append(fig)

    return hemoglobin_figures


# =========================
# LOAD DATA
# =========================
print(f'Participant ID: {ID}')
print(f'Reading files from: {grip_directory}')

participant_information = load_participant_information(
    participants_excel_path=participants_excel_path,
    ID=ID
)

training_sets = load_all_training_sets(
    grip_directory=grip_directory,
    number_of_training_sets=number_of_training_sets
)


# =========================
# CALCULATE SPATIAL ERROR
# =========================
spatial_error_sets = calculate_spatial_error_for_all_sets(training_sets)

spatial_error_summary_df = calculate_spatial_error_summary(spatial_error_sets)

spatial_error_data = create_spatial_error_data_for_javascript(spatial_error_sets)


# =========================
# LOAD AND ANALYZE PERTURBATIONS
# =========================
upward_trials, downward_trials, isometric_high, isometric_low = load_perturbation_trials(
    grip_directory=grip_directory,
    sampling_frequency=sampling_frequency,
    low_pass_filter_frequency=low_pass_filter_frequency
)

isometric_thresholds = calculate_isometric_thresholds(
    isometric_high=isometric_high,
    isometric_low=isometric_low,
    time_threshold=time_threshold
)

upward_perturbation_results = analyze_perturbation_group(
    trials=upward_trials,
    sd_factor=sd_factor,
    time_window=time_window,
    mean_spatial_error_isometric_trials=isometric_thresholds['mean_high'],
    sd_spatial_error_isometric_trials=isometric_thresholds['sd_high'],
    direction='Upward'
)

downward_perturbation_results = analyze_perturbation_group(
    trials=downward_trials,
    sd_factor=sd_factor,
    time_window=time_window,
    mean_spatial_error_isometric_trials=isometric_thresholds['mean_low'],
    sd_spatial_error_isometric_trials=isometric_thresholds['sd_low'],
    direction='Downward'
)

upward_perturbations_fig = create_perturbation_figure(
    analyzed_trials=upward_perturbation_results,
    figure_title=f'{ID} - Upward Perturbations'
)

downward_perturbations_fig = create_perturbation_figure(
    analyzed_trials=downward_perturbation_results,
    figure_title=f'{ID} - Downward Perturbations'
)

perturbation_force_target_data = create_perturbation_force_target_data_for_javascript(
    upward_perturbation_results=upward_perturbation_results,
    downward_perturbation_results=downward_perturbation_results
)


# =========================
# LOAD AND CREATE HEMOGLOBIN FIGURES
# =========================
hemoglobin_data, hemoglobin_fs, hemoglobin_training_sets = load_hemoglobin_training_sets(
    brain_directory=brain_directory,
    artinis_file_name=artinis_file_name
)

hemoglobin_training_figures = create_all_hemoglobin_training_figures(
    training_sets=training_sets,
    hemoglobin_training_sets=hemoglobin_training_sets,
    fs=hemoglobin_fs
)


# =========================
# CREATE HTML REPORT
# =========================
html_parts = []

html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{escape(ID)} - Prequalification Report</title>

    <style>
        html {{
            scroll-behavior: smooth;
        }}

        body {{
            font-family: Georgia, serif;
            margin: 40px;
            background-color: #f7f7f7;
            color: #222;
        }}

        h1 {{
            font-size: 34px;
            margin-bottom: 5px;
        }}

        h2 {{
            font-size: 26px;
            margin-top: 10px;
        }}

        h3 {{
            font-size: 22px;
            margin-top: 20px;
        }}

        .subtitle {{
            font-size: 16px;
            color: #555;
            margin-bottom: 30px;
        }}

        .card {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .toc {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .toc h2 {{
            margin-top: 0;
        }}

        .toc ul {{
            margin: 0;
            padding-left: 22px;
            font-size: 18px;
            line-height: 1.9;
        }}

        .toc a {{
            color: #1f4e79;
            text-decoration: none;
            font-weight: bold;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        .figure-section {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .new-report-page {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            page-break-before: always;
        }}

        .brain-figure-subsection {{
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 30px;
        }}

        .small-text {{
            font-size: 14px;
            color: #666;
        }}

        .back-to-top {{
            font-size: 14px;
            margin-top: 10px;
        }}

        .back-to-top a {{
            color: #1f4e79;
            text-decoration: none;
            font-weight: bold;
        }}

        .back-to-top a:hover {{
            text-decoration: underline;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0, 0, 0, 0.55);
        }}

        .modal-content {{
            background-color: white;
            margin: 4% auto;
            padding: 25px;
            border-radius: 12px;
            width: 90%;
            max-width: 1200px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }}

        .close-button {{
            color: #555;
            float: right;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
        }}

        .close-button:hover {{
            color: black;
        }}
    </style>
</head>

<body id="top">

    <h1>Prequalification Report</h1>

    <div class="subtitle">
        Participant ID: <strong>{escape(ID)}</strong><br>
        Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>

    <section class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#basic-information">Basic Information</a></li>
            <li><a href="#training-section">Training Sets Overview</a></li>
            <li><a href="#spatial-error-summary-section">Spatial Error Summary</a></li>
            <li><a href="#upward-perturbations-section">Upward Perturbations</a></li>
            <li><a href="#downward-perturbations-section">Downward Perturbations</a></li>
            <li><a href="#hemoglobin-training-section">Hemoglobin activity during training sets</a></li>
        </ul>
    </section>

    <section id="basic-information" class="card">
        <h2>Basic Information</h2>
        {create_participant_information_html(participant_information)}
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
""")


# =========================
# PAGE 1: TRAINING SETS FIGURE
# =========================
training_overview_fig = create_training_overview_figure(training_sets)

html_parts.append("""
    <section id="training-section" class="figure-section">
        <h2>Training Sets Overview</h2>
        <p class="small-text">
            Click on any training set to open the detailed spatial error graph for that set.
        </p>
""")

training_fig_html = pio.to_html(
    training_overview_fig,
    full_html=False,
    include_plotlyjs=True,
    div_id='training_overview_plot',
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'responsive': True
    }
)

html_parts.append(training_fig_html)

html_parts.append("""
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
""")


# =========================
# PAGE 2: SPATIAL ERROR SUMMARY
# =========================
spatial_error_summary_fig = create_spatial_error_summary_figure(spatial_error_summary_df)

add_plotly_figure_to_report(
    html_parts=html_parts,
    title='Spatial Error Summary',
    fig=spatial_error_summary_fig,
    div_id='spatial_error_summary_plot',
    section_id='spatial-error-summary-section',
    include_plotlyjs=False,
    page_class='new-report-page'
)


# =========================
# PAGE 3: UPWARD PERTURBATIONS
# =========================
html_parts.append("""
    <section id="upward-perturbations-section" class="new-report-page">
        <h2>Upward Perturbations</h2>
        <p class="small-text">
            Row 1 shows the pre-training perturbations and Row 2 shows the post-training perturbations.
            Columns 1 to 3 correspond to trials 1 to 3.
            The dotted horizontal line is the threshold from the high isometric trial.
            The red vertical line is the adaptation instance.
            Click on any subplot to open the Performance and Target graph for that perturbation.
        </p>
""")

upward_perturbations_fig_html = pio.to_html(
    upward_perturbations_fig,
    full_html=False,
    include_plotlyjs=False,
    div_id='upward_perturbations_plot',
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'responsive': True
    }
)

html_parts.append(upward_perturbations_fig_html)

html_parts.append("""
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
""")


# =========================
# PAGE 4: DOWNWARD PERTURBATIONS
# =========================
html_parts.append("""
    <section id="downward-perturbations-section" class="new-report-page">
        <h2>Downward Perturbations</h2>
        <p class="small-text">
            Row 1 shows the pre-training perturbations and Row 2 shows the post-training perturbations.
            Columns 1 to 3 correspond to trials 1 to 3.
            The dotted horizontal line is the threshold from the low isometric trial.
            The red vertical line is the adaptation instance.
            Click on any subplot to open the Performance and Target graph for that perturbation.
        </p>
""")

downward_perturbations_fig_html = pio.to_html(
    downward_perturbations_fig,
    full_html=False,
    include_plotlyjs=False,
    div_id='downward_perturbations_plot',
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'responsive': True
    }
)

html_parts.append(downward_perturbations_fig_html)

html_parts.append("""
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
""")


# =========================
# PAGE 5: HEMOGLOBIN ACTIVITY DURING TRAINING SETS
# =========================
html_parts.append("""
    <section id="hemoglobin-training-section" class="new-report-page">
        <h2>Hemoglobin activity during training sets</h2>
        <p class="small-text">
            Each graph shows the six filtered O2Hb channels together with the force tracking signal.
            The first 10 seconds are shown before the force-tracking task starts.
            The vertical dashed line indicates the start of the training set.
        </p>
""")

for i, fig in enumerate(hemoglobin_training_figures, start=1):
    html_parts.append(f"""
        <div class="brain-figure-subsection">
            <h3>Training Set {i}</h3>
    """)

    fig_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=f'hemoglobin_training_set_{i}_plot',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'responsive': True
        }
    )

    html_parts.append(fig_html)

    html_parts.append("""
        </div>
    """)

html_parts.append("""
        <p class="back-to-top"><a href="#top">Back to top</a></p>
    </section>
""")


# =========================
# HIDDEN SPATIAL ERROR POPUP
# =========================
html_parts.append("""
    <div id="spatial-error-modal" class="modal">
        <div class="modal-content">
            <span id="close-spatial-error-modal" class="close-button">&times;</span>
            <h2 id="spatial-error-title">Spatial Error</h2>
            <div id="spatial-error-plot"></div>
        </div>
    </div>
""")


# =========================
# HIDDEN PERTURBATION PERFORMANCE/TARGET POPUP
# =========================
html_parts.append("""
    <div id="perturbation-force-target-modal" class="modal">
        <div class="modal-content">
            <span id="close-perturbation-force-target-modal" class="close-button">&times;</span>
            <h2 id="perturbation-force-target-title">Perturbation Force and Target</h2>
            <div id="perturbation-force-target-plot"></div>
        </div>
    </div>
""")


# =========================
# JAVASCRIPT FOR CLICK POPUP
# =========================
html_parts.append(create_popup_javascript(spatial_error_data))

html_parts.append(
    create_perturbation_force_target_popup_javascript(
        perturbation_force_target_data=perturbation_force_target_data
    )
)


# =========================
# CLOSE HTML
# =========================
html_parts.append("""
    <div class="card small-text">
        End of report.
        <br>
        <a href="#top">Back to top</a>
    </div>

</body>
</html>
""")


# =========================
# SAVE REPORT
# =========================
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))

print('HTML report saved here:')
print(report_path)
print('The report was not opened automatically.')
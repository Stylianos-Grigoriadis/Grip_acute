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
    r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen\Sine_1'
)

ID = directory.name
grip_directory = directory / 'Grip data'

number_of_training_sets = 10

report_path = directory / f'{ID}_prequalification_report.html'

force_color = 'blue'
target_color = 'red'
spatial_error_color = 'black'
spatial_error_band_color = 'rgba(0, 0, 0, 0.18)'


# =========================
# FUNCTIONS
# =========================
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


# =========================
# LOAD DATA
# =========================
print(f'Participant ID: {ID}')
print(f'Reading files from: {grip_directory}')

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
        </ul>
    </section>

    <section id="basic-information" class="card">
        <h2>Basic Information</h2>
        <p><strong>Participant folder:</strong> {escape(str(directory))}</p>
        <p><strong>Grip data folder:</strong> {escape(str(grip_directory))}</p>
        <p><strong>Number of training sets:</strong> {number_of_training_sets}</p>
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
# JAVASCRIPT FOR CLICK POPUP
# =========================
html_parts.append(create_popup_javascript(spatial_error_data))


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
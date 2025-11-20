import numpy as np
# import fathon
# from fathon import fathonUtils as fu
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import colorednoise as cn
import random
from scipy.optimize import curve_fit
from scipy.signal import decimate
from scipy.signal import welch
from scipy import stats
import itertools
from itertools import chain
from scipy.stats import pearsonr
from scipy.signal import welch
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
import polars as pl


def Ent_Ap(data, dim, r):
    """
    Ent_Ap20120321
      data : time-series data
      dim : embedded dimension
      r : tolerance (typically 0.2)

      Changes in version 1
          Ver 0 had a minor error in the final step of calculating ApEn
          because it took logarithm after summation of phi's.
          In Ver 1, I restored the definition according to original paper's
          definition, to be consistent with most of the work in the
          literature. Note that this definition won't work for Sample
          Entropy which doesn't count self-matching case, because the count
          can be zero and logarithm can fail.

      *NOTE: This code is faster and gives the same result as ApEn =
             ApEnt(data,m,R) created by John McCamley in June of 2015.
             -Will Denton

    ---------------------------------------------------------------------
    coded by Kijoon Lee,  kjlee@ntu.edu.sg
    Ver 0 : Aug 4th, 2011
    Ver 1 : Mar 21st, 2012
    ---------------------------------------------------------------------
    """

    r = r * np.std(data)
    N = len(data)
    phim = np.zeros(2)
    for j in range(2):
        m = dim + j
        phi = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))
        for i in range(m):
            data_mat[:, i] = data[i:N - m + i + 1]
        for i in range(N - m + 1):
            temp_mat = np.abs(data_mat - data_mat[i, :])
            AorB = np.unique(np.where(temp_mat > r)[0])
            AorB = len(temp_mat) - len(AorB)
            phi[i] = AorB / (N - m + 1)
        phim[j] = np.sum(np.log(phi)) / (N - m + 1)
    AE = phim[0] - phim[1]
    return AE

def Ent_Samp(data, m, r):
    """
    function SE = Ent_Samp20200723(data,m,r)
    SE = Ent_Samp20200723(data,m,R) Returns the sample entropy value.
    inputs - data, single column time seres
            - m, length of vectors to be compared
            - r, radius for accepting matches (as a proportion of the
              standard deviation)

    output - SE, sample entropy
    Remarks
    - This code finds the sample entropy of a data series using the method
      described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
      time-series analysis using approximate entropy and sample entropy."
      Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.
    - m is generally recommendation as 2
    - R is generally recommendation as 0.2
    May 2016 - Modified by John McCamley, unonbcf@unomaha.edu
             - This is a faster version of the previous code.
    May 2019 - Modified by Will Denton
             - Added code to check version number in relation to a server
               and to automatically update the code.
    Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Removed the code that automatically checks for updates and
               keeps a version history.
    Define r as R times the standard deviation
    """
    R = r * np.std(data)
    N = len(data)

    data = np.array(data)

    dij = np.zeros((N - m, m + 1))
    dj = np.zeros((N - m, 1))
    dj1 = np.zeros((N - m, 1))
    Bm = np.zeros((N - m, 1))
    Am = np.zeros((N - m, 1))

    for i in range(N - m):
        for k in range(m + 1):
            dij[:, k] = np.abs(data[k:N - m + k] - data[i + k])
        dj = np.max(dij[:, 0:m], axis=1)
        dj1 = np.max(dij, axis=1)
        d = np.where(dj <= R)
        d1 = np.where(dj1 <= R)
        nm = d[0].shape[0] - 1  # subtract the self match
        Bm[i] = nm / (N - m)
        nm1 = d1[0].shape[0] - 1  # subtract the self match
        Am[i] = nm1 / (N - m)

    Bmr = np.sum(Bm) / (N - m)
    Amr = np.sum(Am) / (N - m)

    return -np.log(Amr / Bmr)

def Perc(signal , upper_lim, lower_lim):
    """This function takes a signal as a np.array and turns it as values from upper_lim to lower_lim"""
    if np.min(signal) < 0:
        signal = signal - np.min(signal)
    signal = 100 * signal / np.max(signal)
    min_val = signal.min()
    max_val = signal.max()
    signal = (signal - min_val) / (max_val - min_val)
    new_range = upper_lim - lower_lim
    signal = signal * new_range + lower_lim
    return signal

def read_kinvent(path):
    """This funcion reads the Kinvent csv file for the grip"""
    df = pd.read_csv(path, header=None, delimiter=';')
    index = []
    for i, string in enumerate(df[0]):
        if 'Repetition: ' in string:
            index.append(i)
    print(index)
    df_set_1 = pd.read_csv(path, skiprows=2, nrows=index[1] - 3)
    df_set_2 = pd.read_csv(path, skiprows=index[1] + 2, nrows=index[2] - index[1] - 3)
    df_set_3 = pd.read_csv(path, skiprows=index[2] + 2, nrows=index[3] - index[2] - 3)
    df_set_4 = pd.read_csv(path, skiprows=index[3] + 2, nrows=index[4] - index[3] - 3)
    df_set_5 = pd.read_csv(path, skiprows=index[4] + 2, nrows=index[5] - index[4] - 3)
    df_set_6 = pd.read_csv(path, skiprows=index[5] + 2, nrows=index[6] - index[5] - 3)
    df_set_7 = pd.read_csv(path, skiprows=index[6] + 2, nrows=index[7] - index[6] - 3)
    df_set_8 = pd.read_csv(path, skiprows=index[7] + 2, nrows=index[8] - index[7] - 3)
    df_set_9 = pd.read_csv(path, skiprows=index[8] + 2, nrows=index[9] - index[8] - 3)
    df_set_10 = pd.read_csv(path, skiprows=index[9] + 2)

    return (df_set_1,
            df_set_2,
            df_set_3,
            df_set_4,
            df_set_5,
            df_set_6,
            df_set_7,
            df_set_8,
            df_set_9,
            df_set_10)

def sine_signal_generator(Number_of_data_points, frequency, upper_lim, lower_lim):

    x = np.arange(0, Number_of_data_points)
    signal = np.sin(x*frequency)
    signal = Perc(signal, upper_lim, lower_lim)

    # time = np.arange(0, Total_Time, Total_Time / Number_of_data_points)
    # return signal, time
    return signal

def isometric_generator_with_reps(Number_of_data_points,value):
    reps_in_set = 20
    total_reps = Number_of_data_points/reps_in_set
    targets_in_each_rep = Number_of_data_points/total_reps
    array_force = np.full(int(targets_in_each_rep/2), value)
    array_zero = np.zeros(int(targets_in_each_rep/2))
    array_single_rep = np.concatenate((array_zero, array_force))
    signal = np.tile(array_single_rep, reps_in_set)
    return signal

def isometric_generator_single_rep(Number_of_data_points,value):
    signal = np.full(Number_of_data_points, value)
    return signal

def create_txt_file(signal, name, path):
    "This Function takes a np.array and turns it into a txt file so that it can be used in the KINVENT grip game"
    element = ''
    for i in signal:
        element = element + str(i) + ','
    element = element[:-1]
    list_to_save = [element]
    df = pd.DataFrame(list_to_save)
    df.to_csv(rf'{path}\{name}.txt',header=False, index=False, sep=' ')

def make_it_random(up_1, up_2, up_3, down_1, down_2, down_3):
    list1 = [up_1, up_2, up_3, down_1, down_2, down_3]
    random.shuffle(list1)

    return list1

def perturbation_both_force(up_1, up_2, up_3, down_1, down_2, down_3, step_1, step_2, step_3, data_num):
    dat_for_each_pert = int(data_num/12)

    baseline = np.zeros(dat_for_each_pert)
    pert_up_1 = np.full(dat_for_each_pert, up_1)
    pert_up_2 = np.full(dat_for_each_pert, up_2)
    pert_up_3 = np.full(dat_for_each_pert, up_3)
    pert_down_1 = np.full(dat_for_each_pert, down_1)
    pert_down_2 = np.full(dat_for_each_pert, down_2)
    pert_down_3 = np.full(dat_for_each_pert, down_3)
    pert_step_1 = np.full(int(dat_for_each_pert/3), step_1)
    pert_step_2 = np.full(int(dat_for_each_pert/3), step_2)
    pert_step_3 = np.full(int(dat_for_each_pert/3), step_3)
    pert_down_whole_1 = np.concatenate((pert_step_1, pert_down_1))
    pert_down_whole_2 = np.concatenate((pert_step_2, pert_down_2))
    pert_down_whole_3 = np.concatenate((pert_step_3, pert_down_3))

    overall_list = make_it_random(pert_up_1, pert_up_2, pert_up_3, pert_down_whole_1, pert_down_whole_2, pert_down_whole_3)

    final_pert = np.concatenate((baseline, overall_list[0],
                                 baseline, overall_list[1],
                                 baseline, overall_list[2],
                                 baseline, overall_list[3],
                                 baseline, overall_list[4],
                                 baseline, overall_list[5]))
    return final_pert

def total_force(signal):
    total = np.sum(signal)
    return total

def synchronization_of_Time_and_ClosestSampleTime_Stylianos(df, Targets_N):
    """ This function takes a dataframe and converts it so that the Time column and the ClosestSampleTime column
        are matched. This is a method to synchronize the two time stamps
    Parameters
    Input
            df          :   the Dataframe
            Target_N    :   the total number of targets
    Output
            new_df      :   the new Dataframe

    """
    # Find the index of the first value where the ClosestSampleTime is equal to Time
    df['Time'] = df['Time'].round(3)
    df['ClosestSampleTime'] = df['ClosestSampleTime'].round(3)

    # Find the value of the column Time with the smallest absolute difference with the first value of ClosestSampleTime
    closest_value = df['Time'].iloc[(df['Time'] - df['ClosestSampleTime'][0]).abs().idxmin()]

    # Find the index of the column Time with the smallest absolute difference with the first value of ClosestSampleTime
    index = df.loc[df['Time'] == closest_value].index[0]

    # Create a list of ClosestSampleTime with the initial value being the value of Time with the smallest difference
    # with the first value of ClosestSampleTime
    initial_value_time = df['Time'][index]
    list_ClosestSampleTime = [initial_value_time]
    for i in range(Targets_N - 1):
        list_ClosestSampleTime.append(round(list_ClosestSampleTime[-1] + 0.3, 3))

    # Create a list of indices of column Time, where the values of list_ClosestSampleTime are equal with the values of
    # column Time
    matching_indices = df.index[df['Time'].isin(list_ClosestSampleTime)].tolist()

    # Create the Performance, Time, and Target lists with the values at the right indices of the initial df
    Performance = [df['Performance'].iloc[i] for i in matching_indices]
    Time = [df['Time'].iloc[i] for i in matching_indices]
    Target = df['Target'].head(len(matching_indices)).tolist()

    # Delete any values from the end of the list_ClosestSampleTime so that all lists Performance, Time, Target and list_ClosestSampleTime
    # have the same length
    list_ClosestSampleTime = list_ClosestSampleTime[:len(Time)]

    # create the new_df
    dist = {'Time': Time, 'Performance': Performance, 'ClosestSampleTime': list_ClosestSampleTime, 'Target': Target}
    new_df = pd.DataFrame(dist)

    return new_df

def synchronization_of_Time_and_ClosestSampleTime_Anestis(df):
    """ This function creates a new dataframe by synchronizing the Time column to the ClosestSampleTime column and then returns a new dataframe with the correct values"""

    # The following lines where added because sometimes the ClosestSampleTime column starts with a negative value.
    # A temporal fix is to make each negative value None and erase it after
    for i in range(len(df['ClosestSampleTime'])):
        if df['ClosestSampleTime'][i] < 0:
            print(f'Negative first value {df.loc[i, "ClosestSampleTime"]}')
            df.loc[i, "ClosestSampleTime"] = None
            df.loc[i, "Target"] = None

    time_index = []
    for i in range(len(df)):
        # Calculate the difference of the element i of the column ClosestSampleTime with every value of the column Time
        closest_index = (df['Time'] - df['ClosestSampleTime'].iloc[i]).abs()
        # Drop the None values of the closest_index so that in the next step the .empty attribute if it has only None values it would show False
        closest_index = closest_index.dropna()

        if not closest_index.empty:
            # Find the index of the minimum difference
            closest_index = closest_index.idxmin()
            # Keep only the index of minimum difference
            time_index.append(closest_index)
    # Create all other columns
    time = df.loc[time_index, 'Time'].to_numpy()
    performance = df.loc[time_index, 'Performance'].to_numpy()
    targets = df['Target'].dropna().to_numpy()
    time_close_to_target = df['ClosestSampleTime'].dropna().to_numpy()


    # Create the dataframe which will be returned afterward.
    dist = {'Indices': time_index,
            'Time': time,
            'Performance': performance,
            'ClosestSampleTime': time_close_to_target,
            'Target': targets}
    new_df = pd.DataFrame(dist)

    return new_df

def isolate_Target(df):

    new_Time = []
    ClosestSampleTime = df['ClosestSampleTime'].dropna().to_list()
    Performance = []
    Target = []
    index_list = []

    Time = df['Time'].to_list()
    # Time = [round(value, 3) for value in Time]
    df['Time'] = df['Time'].round(3)
    print(df['Time'])

    for i in ClosestSampleTime:
        if i in df['Time'].values:
            index = df.index[df['Time'] == i].tolist()[0]
            print(index)
            Performance.append(df['Performance'][index])
            Target.append(df['Target'][index])
            index_list.append(index)
            new_Time.append(i)
    df_targets = pd.DataFrame({'Time': new_Time, 'Performance': Performance, 'ClosestSampleTime': ClosestSampleTime, 'Target': Target})

    return df_targets

def spatial_error(df):
    """ Calculate the spatial error of the Performance and Target
    Parameters
    Input
            df              :   the Dataframe
    Output
            spatial_error   :   the spatial_error between the Performance and Target
    """

    spatial_error = []
    for i in range(len(df['Time'])):
        # Spatial error with absolute values
        spatial_error.append((abs(df['Performance'][i]-df['Target'][i])))
        # Spatial error with algebraic values
        #spatial_error.append((df['Performance'][i]-df['Target'][i]))
    spatial_error = np.array(spatial_error)
    # Removes the np.float64(...) while printing Spatial error
    spatial_error = [float(val) if val is not None else float('nan') for val in spatial_error]
    return spatial_error

def read_my_txt_file(path):
    df = pd.read_csv(path, delimiter=',', decimal='.',header=None)

    signal_list = []
    for i in range(df.shape[1]):
        signal_list.append(df[i][0])
    signal = np.array(signal_list)

    return signal

def asymptotes(df):
    index_where_perturbation_occured = 99
    time = 10
    error = spatial_error(df['Performance'], df['Target'])
    mean = np.mean(error[int(index_where_perturbation_occured/2):index_where_perturbation_occured-1])
    sd = np.std(error[int(index_where_perturbation_occured/2):index_where_perturbation_occured-1])
    error = error[index_where_perturbation_occured:]
    print(len(df['Time']))
    time_for_each_target = time/len(df['Time'])
    print(time_for_each_target)

    index = np.array([i for i in range(len(error))])
    def f(x, a, b, c):
        return a * (b ** x) + c

    popt, _ = curve_fit(f, index, error, bounds=((0, 0, -np.inf), (np.inf, 1, np.inf)), maxfev=30000)
    a, b, c = popt
    print(f'y = {a} * {b}**x + {c}')
    x_line = np.arange(0, len(index), 1)
    y_line = f(x_line, a, b, c)

    plt.plot(x_line, y_line, '--', color='green', label='fit')
    plt.axhline(y=c, c='k', label='Asymptote')
    sd_factor = 1
    c = mean
    plt.axhline(y=c, c='red', label='Mean error before perturbation')
    plt.axhline(y=c + sd_factor * sd, c='red', ls=":", label="sd error before perturbation")
    plt.axhline(y=c - sd_factor * sd, c='red', ls=":")
    plt.scatter(index,error)
    plt.legend()
    plt.show()
    for i in range(len(error)-5):
        if c - sd_factor * sd < error[i] < c + sd_factor * sd and c - sd_factor * sd < error[i+1] < c + sd_factor * sd and c - sd_factor * sd < error[i+sd_factor] < c + sd_factor * sd and c - sd_factor * sd < error[i+3] < c + sd_factor * sd and c - sd_factor * sd < error[i+4] < c + sd_factor * sd:
            print(i)
            adaptation_index = i
            break
    print(f'adaptation index was {adaptation_index}')
    print(f'adaptation time was {adaptation_index*time_for_each_target}')
    dict = {'adaptation_index' : adaptation_index,
            'adaptation_time' : adaptation_index*time_for_each_target}
    return dict

def adaptation_time_using_sd(df, sd_factor, consecutive_values, name, mean_spatial_error_isometric_trials, sd_spatial_error_isometric_trials, plot=False):
    """
    This function returns the time after the perturbation which was needed to adapt to the perturbation
    Parameters
    Input
            df                                      :   The Dataframe
            sd_factor                               :   This will be multiplied with the sd of the error before the perturbation
                                                        and if the error after the is less than the mean + sd*sd_factor and more than
                                                        the mean - sd*sd_factor, the algorithm will consider that the adaptation of the
                                                        perturbation occurred
            consecutive_values                      :   This is how many values the algorithm needs to consider so that it decides that the adaptation occurred.
            name                                    :   The name of the participant and/or the trial
            mean_spatial_error_isometric_trials     :   The average of spatial error which has been previously calculated by isometric trials
            sd_spatial_error_isometric_trials       :   The sd of spatial error which has been previously calculated by isometric trials
            Plot                                    :   Plot the spatial error and the time of adaptation (default value False)

    Output
            time_of_adaptation                      :   The time it took the df['Performance'] to steadily reach df['Target']. This
                                                        number corresponds to the first value of time at which for the next X consecutive_values
                                                        the spatial error was lower than the average +- (sd * sd_factor)
    """
    # First synchronize the Time and ClosestSampleTime columns and create a new df with
    # only the synchronized values
    df = synchronization_of_Time_and_ClosestSampleTime_Anestis(df)
    # print(df['Target'])

    perturbation_index = df[df['Target'] != df['Target'].shift(1)].index[1]
    print(perturbation_index)

    # Calculate the spatial error and the average and sd of the spatial error
    # after the first_values
    spatial_er = spatial_error(df)
    # plt.plot(spatial_er)
    # plt.show()

    mean = mean_spatial_error_isometric_trials
    sd_before_perturbation = sd_spatial_error_isometric_trials

    # Create an array with consecutive_values equal number
    consecutive_values_list = np.arange(0,consecutive_values,1)

    # Iterate the spatial error after the perturbation_index to calculate the time of adaptation
    for i in range(len(spatial_er) - consecutive_values+1):
        if i >= perturbation_index:

            if (all(spatial_er[i + j] < mean + sd_before_perturbation * sd_factor for j in consecutive_values_list) and
                all(spatial_er[i + j] > mean - sd_before_perturbation * sd_factor for j in consecutive_values_list)
            ):
                time_of_adaptation = df['Time'][i] - df['Time'][perturbation_index]
                break

    if plot == True:
        try:
            time_of_adaptation
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.axhline(y=mean, c='k', label = 'Average')
            plt.axhline(y=mean + sd_before_perturbation*sd_factor, c='k', ls=":", label='std')
            plt.axhline(y=mean - sd_before_perturbation*sd_factor, c='k', ls=":")
            plt.axvline(x=df['Time'][perturbation_index] + time_of_adaptation, lw=3, c='red', label='Adaptation instance')
            plt.axvline(x=df['Time'][perturbation_index], linestyle='--', c='gray', label='Perturbation instance')

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name} Spatial Error\ntime to adapt: {round(time_of_adaptation,3)} sec')
            plt.show()
        except NameError:
            print(f"No adaptation was evident for {name}")

    try:
        return time_of_adaptation
    except NameError:
        print(f"No adaptation was evident for {name}")

def single_perturbation_generator(baseline, perturbation, data_num):
    baseline_array = np.full(int(data_num/2), baseline)
    perturbation_array = np.full(int(data_num/2), perturbation)
    final_pert = np.concatenate((baseline_array, perturbation_array))

    return final_pert

def isometric_min_max(MVC):
    sd = 10
    iso_90 = 90
    iso_80 = 80
    iso_70 = 70
    iso_60 = 60
    iso_50 = 50
    iso_40 = 40
    iso_30 = 30
    iso_20 = 20
    iso_15 = 15
    iso_10 = 10
    iso_5 = 5
    iso_2_half = 2.5

    iso_90_perc = MVC * iso_90 / 100
    iso_80_perc = MVC * iso_80 / 100
    iso_70_perc = MVC * iso_70 / 100
    iso_60_perc = MVC * iso_60 / 100
    iso_50_perc = MVC * iso_50 / 100
    iso_40_perc = MVC * iso_40 / 100
    iso_30_perc = MVC * iso_30 / 100
    iso_20_perc = MVC * iso_20 / 100
    iso_15_perc = MVC * iso_15 / 100
    iso_10_perc = MVC * iso_10 / 100
    iso_5_perc = MVC * iso_5 / 100
    iso_2_half_perc = MVC * iso_2_half / 100

    iso_90_min = (iso_90 - sd) * MVC / 100
    iso_80_min = (iso_80 - sd) * MVC / 100
    iso_70_min = (iso_70 - sd) * MVC / 100
    iso_60_min = (iso_60 - sd) * MVC / 100
    iso_50_min = (iso_50 - sd) * MVC / 100
    iso_40_min = (iso_40 - sd) * MVC / 100
    iso_30_min = (iso_30 - sd) * MVC / 100
    iso_20_min = (iso_20 - sd) * MVC / 100
    iso_15_min = (iso_15 - sd) * MVC / 100
    iso_10_min = (iso_10 - sd) * MVC / 100
    iso_5_min = (iso_5 - iso_5) * MVC / 100
    iso_2_half_min = (iso_2_half - iso_2_half) * MVC / 100

    iso_90_max = (iso_90 + sd) * MVC / 100
    iso_80_max = (iso_80 + sd) * MVC / 100
    iso_70_max = (iso_70 + sd) * MVC / 100
    iso_60_max = (iso_60 + sd) * MVC / 100
    iso_50_max = (iso_50 + sd) * MVC / 100
    iso_40_max = (iso_40 + sd) * MVC / 100
    iso_30_max = (iso_30 + sd) * MVC / 100
    iso_20_max = (iso_20 + sd) * MVC / 100
    iso_15_max = (iso_15 + sd) * MVC / 100
    iso_10_max = (iso_10 + sd) * MVC / 100
    iso_5_max = (iso_5 + iso_5) * MVC / 100
    iso_2_half_max = (iso_2_half + iso_2_half) * MVC / 100


    print(f"For 90% of MVC ({iso_90_perc}) the min values is {iso_90_min} and the max values is {iso_90_max}")
    print(f"For 80% of MVC ({iso_80_perc}) the min values is {iso_80_min} and the max values is {iso_80_max}")
    print(f"For 70% of MVC ({iso_70_perc}) the min values is {iso_70_min} and the max values is {iso_70_max}")
    print(f"For 60% of MVC ({iso_60_perc}) the min values is {iso_60_min} and the max values is {iso_60_max}")
    print(f"For 50% of MVC ({iso_50_perc}) the min values is {iso_50_min} and the max values is {iso_50_max}")
    print(f"For 40% of MVC ({iso_40_perc}) the min values is {iso_40_min} and the max values is {iso_40_max}")
    print(f"For 30% of MVC ({iso_30_perc}) the min values is {iso_30_min} and the max values is {iso_30_max}")
    print(f"For 90% of MVC ({iso_20_perc}) the min values is {iso_20_min} and the max values is {iso_20_max}")
    print(f"For 15% of MVC ({iso_15_perc}) the min values is {iso_15_min} and the max values is {iso_15_max}")
    print(f"For 10% of MVC ({iso_10_perc}) the min values is {iso_10_min} and the max values is {iso_10_max}")
    print(f"For 5% of MVC ({iso_5_perc}) the min values is {iso_5_min} and the max values is {iso_5_max}")
    print(f"For 2.5% of MVC ({iso_2_half_perc}) the min values is {iso_2_half_min} and the max values is {iso_2_half_max}")

def signal_from_min_to_max(signal,max):
    ''' Where:
                signal: is the signal I want to change
                max:    is the max force I inserted into Kinvent app
                '''
    signal = np.array(signal)
    signal = signal * max / 100
    return signal

def add_generated_signal(kinvent_path, generated_signal_path, max_force):
    df_kinvent = pd.read_csv(kinvent_path, skiprows=2)
    df_kinvent_no_zeros = isolate_Target(df_kinvent)
    length_kinvent = len(df_kinvent_no_zeros['Target'])

    generated_signal = read_my_txt_file(generated_signal_path)
    print(generated_signal)
    generated_signal = signal_from_min_to_max(generated_signal, max_force)

    length_generated_signal = len(generated_signal)
    length_erase = length_generated_signal - length_kinvent

    length_generated_signal_erase = length_erase//2 +1
    length_generated_signal_start = length_generated_signal_erase
    print(f'reminder {length_erase % 2}')
    length_generated_signal_end = length_generated_signal - length_generated_signal_erase + length_erase % 2




    print(rf"length_generated_signal_start: {length_generated_signal_start}")
    print(rf"length_generated_signal_end: {length_generated_signal_end}")

    print(rf"length_kinvent: {length_kinvent}")
    print(rf"length_generated_signal_before: {length_generated_signal}")
    print(rf"length_erase: {length_erase}")

    generated_signal = generated_signal[length_generated_signal_start:length_generated_signal_end]
    print(rf"generated_signal: {len(generated_signal)}")

    df_kinvent_no_zeros['Generated_Signal'] = generated_signal

    return df_kinvent_no_zeros

def down_sampling(df, f_out, f_in):
    """
    Parameters
    In
                df:     the dataframe to be downsampled
                f_out:  the frequency I want
                f_in:   the frequency the df has
    Out
                df_downsampled:     the df with downsampled the 'Time' and 'Performance' columns
                                    while 'ClosestSampleTime' and 'Target' were left intact

    """
    factor = int(f_in/f_out)

    df_downsampled_first_two_cols = df[['Time', 'Performance']].iloc[::factor].reset_index(drop=True)
    df_remaining_cols = df[['ClosestSampleTime', 'Target']]
    df_downsampled = pd.concat([df_downsampled_first_two_cols, df_remaining_cols], axis=1)

    return df_downsampled

def outputs(white, pink, sine):
    white_average = np.mean(white)
    pink_average = np.mean(pink)
    sine_average = np.mean(sine)
    list_average = [white_average, pink_average, sine_average]

    white_std = np.std(white)
    pink_std = np.std(pink)
    sine_std = np.std(sine)
    list_std = [white_std, pink_std, sine_std]

    x_axis_white = np.linspace(0, 30, len(white))
    x_axis_pink = np.linspace(0, 30, len(pink))
    x_axis_sine = np.linspace(0, 30, len(sine))

    white_total_load = np.trapz(white, x_axis_white)
    pink_total_load = np.trapz(pink, x_axis_pink)
    sine_total_load = np.trapz(sine, x_axis_sine)
    list_total_load = [white_total_load, pink_total_load, sine_total_load]

    white_dfa = dfa(white)
    pink_dfa = dfa(pink)
    sine_dfa = dfa(sine)
    list_dfa = [white_dfa, pink_dfa, sine_dfa]

    white_SaEn = Ent_Samp(white, 2, 0.2)
    pink_SaEn = Ent_Samp(pink, 2, 0.2)
    sine_SaEn = Ent_Samp(sine, 2, 0.2)
    list_SaEn = [white_SaEn, pink_SaEn, sine_SaEn]

    white_slope, _, _, _, _, _, _, _ = quality_assessment_of_temporal_structure_FFT_method(white)
    pink_slope, _, _, _, _, _, _, _ = quality_assessment_of_temporal_structure_FFT_method(pink)
    sine_slope, _, _, _, _, _, _, _ = quality_assessment_of_temporal_structure_FFT_method(sine)
    list_slope = [round(white_slope, 2), round(pink_slope, 2), round(sine_slope, 2)]
    dist = {'Signals': ['White', 'Pink', 'Sine'],
            'Average': list_average,
            'std': list_std,
            'Total_load': list_total_load,
            'DFA': list_dfa,
            'SaEn': list_SaEn,
            'Slope': list_slope
            }
    df = pd.DataFrame(dist)
    print(df)

def z_transform(signal, desired_sd, desired_average):
    average = np.mean(signal)
    sd = np.std(signal)
    standarized_signal = (signal - average) / sd
    transformed_signal = standarized_signal * desired_sd + desired_average

    return transformed_signal

def sine_wave_signal_creation(N, Number_of_periods, desired_sd, desired_average):
    frequency = Number_of_periods * 2
    t = np.linspace(0, 1, N)
    sine_wave = np.sin(np.pi * frequency * t)
    sine_wave = z_transform(sine_wave, desired_sd, desired_average)

    return sine_wave

def quality_assessment_of_temporal_structure_FFT_method(signal):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1/0.01)  # Frequency bins

    # Keep only the positive frequencies
    positive_freqs = frequencies[1:len(frequencies) // 2]  # Skip the zero frequency
    positive_magnitude = fft_magnitude[1:len(frequencies) // 2]  # Skip the zero frequency

     # Figure of Frequincies vs Magnitude
    plt.figure(figsize=(10,6))
    plt.plot(positive_freqs, positive_magnitude)
    # plt.title(f'{name}\nFFT of Sine Wave')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

    positive_freqs_log = np.log10(positive_freqs[positive_freqs > 0])
    positive_magnitude_log = np.log10(positive_magnitude[positive_freqs > 0])

    r, p = pearsonr(positive_freqs_log, positive_magnitude_log)

    # Perform linear regression (best fit) to assess the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(positive_freqs_log, positive_magnitude_log)
    # print(f'r_value = {r_value}')
    # print(f'p_value = {p_value}')

    # Plot the log-log results
    plt.figure(figsize=(10,6))
    plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    plt.title(f'Log-Log Plot of FFT (Frequency vs Magnitude)')
    plt.xlabel('Log(Frequency) (Hz)')
    plt.ylabel('Log(Magnitude)')
    plt.legend()
    plt.grid()
    plt.show()

    return slope, positive_freqs_log, positive_magnitude_log, intercept, r, p, positive_freqs, positive_magnitude

def one_pink_signal_from_several(num_signals, num_points, desired_sd, desired_average):
    one_pink_signal = []
    for i in range(num_signals):
        one_pink_signal.append(pink_noise_signal_creation_using_cn(num_points, desired_sd, desired_average))

    flattened_list = list(chain.from_iterable(one_pink_signal))

    return flattened_list

def one_white_signal_from_several(num_signals, num_points, desired_sd, desired_average):
    one_white_signal = []
    for i in range(num_signals):
        one_white_signal.append(white_noise_signal_creation(num_points, desired_sd, desired_average))

    flattened_list = list(chain.from_iterable(one_white_signal))

    return flattened_list

def one_sine_signal_from_several(num_signals, num_points, desired_sd, desired_average, num_periods):
    one_sine_signal = []
    for i in range(num_signals):
        one_sine_signal.append(sine_wave_signal_creation(num_points, desired_sd, desired_average, num_periods))

    flattened_list = list(chain.from_iterable(one_sine_signal))

    return flattened_list

def integrate_signal(signal):
    """
    Integrates the input signal to compute the cumulative sum
    after subtracting the mean of the signal.
    Parameters:
    signal (numpy array): Input time series signal
    Returns:
    numpy array: Integrated (cumulative sum) signal
    """
    # Compute the mean of the signal
    mean_signal = np.mean(signal)
    # Subtract the mean from the signal
    detrended_signal = signal - mean_signal
    # Compute the cumulative sum (integrated series)
    integrated_signal = np.cumsum(detrended_signal)
    return integrated_signal

def moving_average(data):
    series = pd.Series(data)
    moving_avg = series.rolling(window=5).mean()
    return moving_avg

def perturbation_single_trial_with_random_change(Number_of_data_points, starting_point, ending_point):
    """This function creates a perturbation from the starting point to the ending point, this happens at a random moment
    from the 40% to the 60% of the duration of the perturbation"""

    shift_amount = float(ending_point - starting_point)
    signal = np.full(Number_of_data_points, starting_point)
    low_limit = Number_of_data_points * 0.4
    high_limit = Number_of_data_points * 0.6 + 1
    random_index = np.random.randint(low_limit, high_limit)
    signal[random_index:] = signal[random_index:] + shift_amount

    return signal

def pink_noise_signal_creation_using_FFT_method(N, desired_sd, desired_average):
    pink = False
    iterations = 0
    while pink == False:

        pink_noise = cn.powerlaw_psd_gaussian(1, N)
        pink_noise = z_transform(pink_noise, desired_sd, desired_average)
        slope, positive_freqs_log, positive_magnitude_log, intercept, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(
            pink_noise)

        if (round(slope, 2) == -0.5) and (p < 0.05) and (r <= -0.7) and (np.all(pink_noise >= 0)) and (np.all(pink_noise <= 100)):
            #
            # Figure of Frequincies vs Magnitude
            # plt.figure(figsize=(10,6))
            # plt.plot(positive_freqs, positive_magnitude)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude')
            # plt.grid()
            # plt.show()
            #
            # plt.figure(figsize=(10, 6))
            # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
            # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept,
            #          label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
            # plt.title(f'Pink noise\nLog-Log Plot of FFT (Frequency vs Magnitude)')
            # plt.xlabel('Log(Frequency) (Hz)')
            # plt.ylabel('Log(Magnitude)')
            # plt.legend()
            # plt.grid()
            # plt.show()
            pink = True
        else:
            # print('Not valid pink noise signal')
            iterations += 1
            # print(iterations)

    return pink_noise

def white_noise_signal_creation_using_FFT_method(N, desired_sd, desired_average):
    pink = False
    iterations = 0
    while pink == False:

        white_noise = np.random.rand(N)
        white_noise = z_transform(white_noise, desired_sd, desired_average)
        slope, positive_freqs_log, positive_magnitude_log, intercept, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(
            white_noise)

        if (round(np.abs(slope), 2) == 0) and (np.all(white_noise >= 0)) and (np.all(white_noise <= 100)):
            #
            # Figure of Frequincies vs Magnitude
            # plt.figure(figsize=(10,6))
            # plt.plot(positive_freqs, positive_magnitude)
            # plt.title(f'{name}\nFFT of Sine Wave')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude')
            # plt.grid()
            # plt.show()

            # plt.figure(figsize=(10, 6))
            # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
            # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept,
            #          label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
            # plt.title(f'White Noise\nLog-Log Plot of FFT (Frequency vs Magnitude)')
            # plt.xlabel('Log(Frequency) (Hz)')
            # plt.ylabel('Log(Magnitude)')
            # plt.legend()
            # plt.grid()
            # plt.show()
            pink = True
        else:
            # print('Not valid pink noise signal')
            iterations += 1
            # print(iterations)

    return white_noise

def fgn_sim(n=1000, H=0.7):
    """Create Fractional Gaussian Noise
     Inputs:
            n: Number of data points of the time series. Default is 1000 data points.
            H: Hurst parameter of the time series. Default is 0.7.
     Outputs:
            An array of n data points with variability H
    # =============================================================================
                                ------ EXAMPLE ------

          - Create time series of 1000 datapoints to have an H of 0.7
          n = 1000
          H = 0.7
          dat = fgn_sim(n, H)

          - If you would like to plot the timeseries:
          import matplotlib.pyplot as plt
          plt.plot(dat)
          plt.title(f"Fractional Gaussian Noise (H = {H})")
          plt.xlabel("Time")
          plt.ylabel("Value")
          plt.show()
    # =============================================================================
    """

    # Settings:
    mean = 0
    std = 1

    # Generate Sequence:
    z = np.random.normal(size=2 * n)
    zr = z[:n]
    zi = z[n:]
    zic = -zi
    zi[0] = 0
    zr[0] = zr[0] * np.sqrt(2)
    zi[n - 1] = 0
    zr[n - 1] = zr[n - 1] * np.sqrt(2)
    zr = np.concatenate([zr[:n], zr[n - 2::-1]])
    zi = np.concatenate([zi[:n], zic[n - 2::-1]])
    z = zr + 1j * zi

    k = np.arange(n)
    gammak = (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H)) / 2
    ind = np.concatenate([np.arange(n - 1), [n - 1], np.arange(n - 2, 0, -1)])
    gammak = gammak[ind]  # Circular shift of gammak to match n
    gkFGN0 = np.fft.ifft(gammak)
    gksqrt = np.real(gkFGN0)

    if np.all(gksqrt > 0):
        gksqrt = np.sqrt(gksqrt)
        z = z[:len(gksqrt)] * gksqrt
        z = np.fft.ifft(z)
        z = 0.5 * (n - 1) ** (-0.5) * z
        z = np.real(z[:n])
    else:
        gksqrt = np.zeros_like(gksqrt)
        raise ValueError("Re(gk)-vector not positive")

    # Standardize: (z - np.mean(z)) / np.sqrt(np.var(z))
    ans = std * z + mean
    return ans

def dfa(data, order=1, k=18, plot=False, sc1=4, sc2=4, ax=None, ax_residual=None):
    nmin = sc1
    nmax = len(data) // sc2

    log_min = np.log10(nmin)
    log_max = np.log10(nmax)
    log_scales = np.linspace(log_min, log_max, k)

    scales = np.unique(np.round(10 ** log_scales).astype(int))

    # Check if data is a column vector (2D array with one column)
    if data.shape[0] == 1:
        # Reshape the data to be a column vector
        data = data.reshape(-1, 1)
    else:
        # Data is already a column vector
        data = data

    # =============================================================================
    ##########################   START DFA CALCULATION   ##########################
    # =============================================================================

    # Step 1: Integrate the data
    integrated_data = np.cumsum(data - np.mean(data))

    fluctuation = []

    for scale in scales:
        # Step 2: Divide data into non-overlapping window of size 'scale'
        chunks = len(data) // scale
        ms = 0.0

        for i in range(chunks):
            this_chunk = integrated_data[i * scale:(i + 1) * scale]
            x = np.arange(len(this_chunk))

            # Step 3: Fit polynomial (default is linear, i.e., order=1)
            coeffs = np.polyfit(x, this_chunk, order)
            fit = np.polyval(coeffs, x)

            # Detrend and calculate RMS for the current window
            ms += np.mean((this_chunk - fit) ** 2)

            # Calculate average RMS for this scale
        fluctuation.append(np.sqrt(ms / chunks))

        # Perform linear regression
    alpha, intercept = np.polyfit(np.log2(scales), np.log2(fluctuation), 1)

    # Create a log-log plot to visualize the results
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(scales, fluctuation, marker='o', markerfacecolor='red', markersize=8,
                  linestyle='-', color='black', linewidth=1.7, label=f'Data')

        # Create the fitted line
        fit_line = 2 ** (intercept) * scales ** (alpha)  # since log2(y) = alpha * log2(x) + intercept
        ax.loglog(scales, fit_line, '-', linewidth=3, ls='--', color='blue', label=f'Fit (alpha = {alpha:.3f})')

        ax.set_xlabel('Scale (log)')
        ax.set_ylabel('Fluctuation (log)')
        ax.set_title('esDFA')
        ax.grid(True)
        ax.legend()

        if ax_residual is not None:
            residuals = np.log2(fluctuation) - (alpha * np.log2(scales) + intercept)
            ax_residual.plot(scales, residuals, 'o-', color='purple', markersize=5)
            ax_residual.axhline(0, linestyle='--', color='gray')
            ax_residual.set_xscale('log')
            ax_residual.set_xlabel('Scale (log)')
            ax_residual.set_ylabel('Residuals')
            ax_residual.grid(True)
        plt.show()

    # Return the scales used, fluctuation functions and the alpha value
    # return scales, fluctuation, alpha
    return alpha

def signal_interpolation(signal, step, plot=False):

    signal = np.array(signal)
    x_original = np.linspace(0, len(signal), len(signal))
    y_original = signal
    total_data_points = step * len(signal)
    x_new = np.linspace(x_original[0], x_original[-1], total_data_points)
    y_new = np.interp(x_new, x_original, y_original)

    if plot == True:
        plt.figure(figsize=(8, 4))
        plt.scatter(x_original, y_original, label='Original (100 pts)', s=100)
        plt.scatter(x_new, y_new, label='Upsampled (1000 pts)', s=1)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Signal")
        plt.title("Linear Interpolation (Resampling from 100 → 1000 points)")
        plt.show()

    return y_new

def artinis_read_file(directory, name):
    """
    This function reads the file of an artinis dataset and returns the dataframe with all data and the sampling frequency
    """
    data = pl.read_excel(f"{directory}\\{name}.xlsx")
    data = data.rename({data.columns[1]: "Unnamed: 1"})
    data = data.slice(2)
    sampling_frequency = float(data['Unnamed: 1'][0])
    total_samples = float(data['Unnamed: 1'][2])
    step = 1 / sampling_frequency
    time = np.arange(0, total_samples * step, step)

    column_names = ['Sample number', '[9322] Rx1 - Tx1,Tx2,Tx3  TSI%', '[9322] Rx1 - Tx1,Tx2,Tx3  TSI Fit Factor', '[9323] Rx3 - Tx4,Tx5,Tx6  TSI%', '[9323] Rx3 - Tx4,Tx5,Tx6  TSI Fit Factor', '[9322] Rx1 - Tx1  O2Hb', '[9322] Rx1 - Tx1  HHb', '[9322] Rx1 - Tx2  O2Hb', '[9322] Rx1 - Tx2  HHb', '[9322] Rx1 - Tx3  O2Hb', '[9322] Rx1 - Tx3  HHb', '[9322] Rx2 - Tx1  O2Hb', '[9322] Rx2 - Tx1  HHb', '[9322] Rx2 - Tx2  O2Hb', '[9322] Rx2 - Tx2  HHb', '[9322] Rx2 - Tx3  O2Hb', '[9322] Rx2 - Tx3  HHb', '[9323] Rx3 - Tx4  O2Hb', '[9323] Rx3 - Tx4  HHb', '[9323] Rx3 - Tx5  O2Hb', '[9323] Rx3 - Tx5  HHb', '[9323] Rx3 - Tx6  O2Hb', '[9323] Rx3 - Tx6  HHb', '[9323] Rx4 - Tx4  O2Hb', '[9323] Rx4 - Tx4  HHb', '[9323] Rx4 - Tx5  O2Hb', '[9323] Rx4 - Tx5  HHb', '[9323] Rx4 - Tx6  O2Hb', '[9323] Rx4 - Tx6  HHb', 'Event', 'Event text']
    data = data.slice(63)

    data = data.rename(dict(zip(data.columns, column_names)))
    data = data.with_columns(
        pl.Series("Time", time)
    )

    cols = data.columns
    new_order = [cols[0], "Time"] + cols[1:-1]  # move Time after the first column

    data = data.select(new_order)
    return data, sampling_frequency

def artinis_read_file_10_sets(directory, name):
    """
    This function reads the file of an artinis dataset and returns a list with 10 dataframes with all data and the sampling
    frequency. The data frames are cut 10 seconds before the beginning of the event, and it stops at the end event.
    """
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
    for i, value in enumerate(data['Event']):
        if value != "":
            print(i, value)
            list_indices.append(i)

    numeric_cols = [c for c in column_names if c not in ["Event", "Event text"]]

    data = data.with_columns(
        [pl.col(c).cast(pl.Float64) for c in numeric_cols]
    )

    seconds_to_keep_before_the_trial = 10
    data_points_to_keep_before_the_trial = int(seconds_to_keep_before_the_trial * sampling_frequency)
    training_sets = {}
    for set_num in range(1, 10):  # 1 to 9
        start = list_indices[2 * set_num - 1]  # 1,3,5,...,17
        end = list_indices[2 * set_num]  # 2,4,6,...,18

        # Polars slice: start row, number of rows
        training_sets[f"training_set_{set_num}"] = data.slice(start - data_points_to_keep_before_the_trial, end - start + data_points_to_keep_before_the_trial)
    # 2) Create training_set_with_pert from indices[19] to indices[21]
    start_pert = list_indices[19]
    end_pert = list_indices[21]


    training_set_with_pert = data.slice(start_pert - data_points_to_keep_before_the_trial, end_pert - start_pert + data_points_to_keep_before_the_trial)

    training_set_1 = training_sets['training_set_1']
    training_set_2 = training_sets['training_set_2']
    training_set_3 = training_sets['training_set_3']
    training_set_4 = training_sets['training_set_4']
    training_set_5 = training_sets['training_set_5']
    training_set_6 = training_sets['training_set_6']
    training_set_7 = training_sets['training_set_7']
    training_set_8 = training_sets['training_set_8']
    training_set_9 = training_sets['training_set_9']
    list_training_sets = [training_set_1, training_set_2, training_set_3, training_set_4, training_set_5, training_set_6, training_set_7, training_set_8, training_set_9, training_set_with_pert]
    return list_training_sets, sampling_frequency

def fNIRS_check_quality(y, fs, plot=True):
    """
        Evaluate fNIRS channel quality by detecting a cardiac peak in the PSD.

        This implements a Perdue/Wyser-style check: compute the Welch power spectral
        density (PSD), isolate the cardiac band (0.6–1.8 Hz), fit a Gaussian to the
        PSD (in dB) within that band, and use the fitted peak height (above local
        baseline) as a quality index. A channel is marked "good" if the peak height
        is >= 12 dB.

        Steps
        -----
        1) Remove DC offset from the input signal.
        2) Estimate PSD via Welch with a 10 s segment (nperseg = fs*10).
        3) Keep only 0.6–1.8 Hz (cardiac band).
        4) Fit a Gaussian: P(f) ~= a * exp(-(f - x0)^2 / (2*sigma^2)) + c
           - a     : peak height above baseline (dB)
           - x0    : peak frequency (Hz)
           - sigma : peak width (Hz)
           - c     : baseline level (dB)
           The quality metric is a (height above baseline). If the fit fails,
           falls back to the max PSD (dB) within the cardiac band.
        5) Decide quality: good = (a >= 12 dB).
        6) Optionally plot the PSD, the cardiac band, and the 12 dB threshold.

        Parameters
        ----------
        y : array-like
            1D time series (e.g., HbO) for a single channel.
        fs : float
            Sampling frequency in Hz.
        plot : bool, optional
            If True, show a diagnostic PSD plot.

        Returns
        -------
        good : bool
            True if the channel passes the 12 dB cardiac-peak criterion.
        peak_height : float
            Fitted cardiac peak height above baseline (dB). If the fit failed,
            this is the max PSD (dB) within 0.6–1.8 Hz.

        Notes
        -----
        - Using the fitted amplitude 'a' (peak above baseline) is slightly different
          from thresholding absolute PSD level (a + c); both are reasonable—be consistent.
        - If there are too few frequency bins in 0.6–1.8 Hz (e.g., very short data),
          the function returns (False, NaN) and optionally plots a warning figure.
        """


    # 1) Remove mean (kill 0 Hz/DC)
    y = y - np.mean(y)

    # 2) Power spectral density (Welch’s method)
    nperseg = int(fs * 10)
    f, Pxx = welch(y, fs=fs, nperseg=nperseg)  # 10 s window
    Pxx_dB = 10 * np.log10(Pxx + 1e-20)  # avoid log(0)

    # 3) Focus on 0.6–1.8 Hz (cardiac band)
    mask = (f >= 0.6) & (f <= 1.8)
    f_hr, P_hr = f[mask], Pxx_dB[mask]

    if f_hr.size < 5:  # too few points to fit reliably
        if plot:
            plt.figure(figsize=(5, 3))
            plt.plot(f, Pxx_dB, label='PSD (dB)')
            plt.axvspan(0.6, 1.8, color='orange', alpha=0.2, label='Cardiac range')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.title('Insufficient points in 0.6–1.8 Hz band')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return False, np.nan

    # 4) Fit a Gaussian to the band
    def gaussian(x, a, x0, sigma, c):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

    a0 = np.max(P_hr) - np.min(P_hr)
    x0 = f_hr[np.argmax(P_hr)]
    sigma0 = 0.2
    c0 = np.min(P_hr)
    try:
        popt, _ = curve_fit(gaussian, f_hr, P_hr, p0=[a0, x0, sigma0, c0], maxfev=10000)

        a, x0, sigma, c = popt
        peak_height = a
    except Exception:
        peak_height = np.max(P_hr)  # fallback if fit fails

    # 5) Decide if it’s a good channel (Perdue/Wyser criterion)
    good = peak_height >= 12

    # 6) Optional plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=False, sharey=False)

        # ---------- TOP: full PSD ----------
        ax1.plot(f, Pxx_dB, color='black', lw=1.2, label='PSD (Welch, dB)')
        ax1.axvspan(0.6, 1.8, color='orange', alpha=0.2, label='Cardiac band (0.6–1.8 Hz)')
        baseline = np.min(P_hr)
        ax1.axhline(baseline, color='gray', linestyle='--', label='Baseline')
        ax1.axhline(baseline + 12, color='red', linestyle='--', label='+12 dB threshold')

        # Gaussian fit
        if 'popt' in locals():
            f_fit = np.linspace(0.6, 1.8, 300)
            ax1.plot(f_fit, gaussian(f_fit, *popt),
                     color='red', lw=2, label='Gaussian fit (heart-rate peak)')

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (dB)')
        ax1.set_title('Power Spectral Density and Cardiac Gaussian Fit — Full Range')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ---------- BOTTOM: zoomed view ----------
        ax2.plot(f, Pxx_dB, color='black', lw=1.2, label='PSD (Welch, dB)')
        ax2.axvspan(0.6, 1.8, color='orange', alpha=0.2, label='Cardiac band (0.6–1.8 Hz)')
        ax2.axhline(baseline, color='gray', linestyle='--', label='Baseline')
        ax2.axhline(baseline + 12, color='red', linestyle='--', label='+12 dB threshold')

        if 'popt' in locals():
            ax2.plot(f_fit, gaussian(f_fit, *popt),
                     color='red', lw=2, label='Gaussian fit (heart-rate peak)')

        ax2.set_xlim(0.4, 2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (dB)')
        ax2.set_title('Zoomed View: 0–3 Hz, −40 to −20 dB')
        # ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return good, peak_height

def moving_standard_deviation(signal, time_window, fs, plot=False):
    """
        Calculate the moving (sliding) standard deviation of the signal's first derivative.

        This function estimates how rapidly the input signal changes within short,
        overlapping time windows. It is primarily used for detecting motion artifacts
        in fNIRS signals (e.g., O2Hb or HHb).

        Steps:
        1. Compute the first derivative of the signal to emphasize fast fluctuations.
        2. Slide a window of duration `time_window` seconds through the derivative
           (windows are fully overlapping, step = 1 sample).
        3. For each position, compute the local mean and mean of squares using
           convolution, and derive the local standard deviation as:
               std = sqrt(E[x^2] - (E[x])^2)
        4. Return the resulting moving standard deviation at each sample.

         Parameters
         ----------
         signal : array-like
             Input time series (e.g., O2Hb or HHb).
         time_window : float
             Length of the sliding window in seconds (e.g., 1.0).
         fs : float
             Sampling frequency in Hz.
         plot : bool, optional
             If True, plot the original signal, its derivative, and the moving STD.

        Returns
        -------
        mov_std : ndarray
            Moving standard deviation (same length as signal).
        dy : ndarray
            First derivative of the signal (for inspection).
        w : int
            Window length in samples used for the moving calculation.

        Notes
        -----
        - Windows are fully overlapping (step = 1 sample) to produce a smooth,
          sample-by-sample estimate of local variability.
        - Larger `time_window` values make detection more robust but less sensitive
          to short transients; smaller values increase sensitivity.
    """

    y = np.array(signal)
    dy = np.diff(y, prepend=y[0]) * fs
    w = int(round(time_window * fs))

    if w % 2 == 0:  # make it odd (nicer centering)
        w += 1

    k = np.ones(w) / w
    m = np.convolve(dy, k, mode='same')  # moving mean
    m2 = np.convolve(dy ** 2, k, mode='same')  # moving mean of squares
    mov_std = np.sqrt(np.maximum(m2 - m ** 2, 0))  # std = sqrt(E[x^2] - (E[x])^2)

    if plot:
        plt.plot(np.linspace(0,len(mov_std),len(mov_std)),mov_std, label='mov_std')
        plt.plot(np.linspace(0,len(y),len(y)),y, label='y')
        plt.plot(np.linspace(0,len(dy),len(dy)),dy, label='dy')
        plt.legend()
        plt.show()

    return mov_std, dy, w

def detect_motion_mask_from_movstd(time_window, signal, fs, thresh_z=4, plot=True):
    """
        Detect motion artifacts in a time-series signal using the moving standard deviation.

        This function standardizes the moving standard deviation (computed from the signal's
        derivative) into robust z-scores and flags segments likely affected by motion artifacts.
        Detection is based on how strongly the local variability deviates from the typical
        (median) behavior of the signal.

        Steps
        -----
        1. Compute the moving standard deviation of the signal's derivative using
           `moving_standard_deviation()`.
        2. Standardize it with a robust z-score:
               z = (mov_std - median) / (1.4826 * MAD)
           where MAD is the Median Absolute Deviation.
        3. Flag samples as motion artifacts when z > `thresh_z`.
        4. Optionally, plot the signal and z-scores with shaded motion regions.

        Parameters
        ----------
        time_window : float
            Length of the sliding window in seconds (passed to `moving_standard_deviation`).
        signal : array-like
            Input time-series data (e.g., O2Hb or HHb).
        fs : float
            Sampling frequency in Hz.
        thresh_z : float, optional
            Z-score threshold for detecting motion artifacts (default is 4.0).
        plot : bool, optional
            If True, plot the signal (blue), z-scores (black), threshold line (red),
            and detected motion regions (shaded red).

        Returns
        -------
        mask : ndarray of bool
            Boolean array (same length as signal). True where motion is detected.
        z : ndarray
            Robust z-scores corresponding to each sample.

        Notes
        -----
        - The robust z-score uses the median and MAD, making it resistant to outliers.
        - Higher `thresh_z` values make detection more conservative (fewer artifacts flagged).
        - The plot provides a quick visual check of detection quality.
        """
    mov_std, dy, w = moving_standard_deviation(signal, time_window, fs)

    mov_std = np.array(mov_std)

    med = np.median(mov_std)
    mad = np.median(np.abs(mov_std - med)) + 1e-12

    z = (mov_std - med) / (1.4826 * mad)

    mask = z > thresh_z

    if plot:
        t = np.arange(len(signal)) / fs
        fig, ax1 = plt.subplots(figsize=(9, 4))

        # plot the raw signal (left y-axis)
        ax1.plot(t, signal, color='blue', label='Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # plot z-score (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(t, z, color='black', label='z-score')
        ax2.axhline(thresh_z, color='red', linestyle='--', lw=1)
        ax2.set_ylabel('Robust z-score', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # mark motion regions
        for i in range(len(mask)):
            if mask[i]:
                ax1.axvspan(t[i], t[i], color='tomato', alpha=0.3)

        plt.title('Motion detection (red = motion)')
        plt.tight_layout()
        plt.show()

    return mask, z

def mask_to_segments(mask, n, fs, z, signal, thresh_z=4, plot=False):
    segs = []
    on = False
    for i, m in enumerate(mask):
        if m and not on:
            s = i
            on = True
        elif not m and on:
            segs.append((s, i-1))
            on = False
    if on:
        segs.append((s, len(mask)-1))
    if not segs:
        return []
    else:
        cleaned = postprocess_segments(segs, n, fs, min_len_sec=0.15, pad_sec=0.20)
        if plot:
            t = np.arange(len(signal)) / fs
            fig, ax1 = plt.subplots(figsize=(9, 4))

            # --- Plot the raw signal (left y-axis) ---
            ax1.plot(t, signal, color='blue', label='Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Signal', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # --- Plot z-score (right y-axis) ---
            ax2 = ax1.twinx()
            ax2.plot(t, z, color='black', label='z-score')
            ax2.axhline(thresh_z, color='red', linestyle='--', lw=1)
            ax2.set_ylabel('Robust z-score', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # --- Mark raw motion points (thin red) ---
            for i in range(len(mask)):
                if mask[i]:
                    ax1.axvspan(t[i], t[i], color='tomato', alpha=0.3)

            # --- Mark postprocessed motion segments (thicker dark red) ---
            for start, end in cleaned:
                ax1.axvspan(t[start], t[end], color='darkred', alpha=0.4)

            plt.title('Motion detection (red = raw, dark red = cleaned)')
            plt.tight_layout()
            plt.show()

    return cleaned

def postprocess_segments(segs, n, fs, min_len_sec=0.15, pad_sec=0.20):
    """
    This function goes through all detected motion segments and performs
    three post-processing steps: padding, merging, and cleaning.
    First, each segment is extended slightly before and after to capture
    the full motion period. Then overlapping or adjacent segments are merged
    into one continuous block. Finally, very short segments (shorter than
    `min_len_sec`) are removed to avoid false detections.

    Parameters
    ----------
    segs : list of tuples
        List of (start_index, end_index) segments (from mask_to_segments()).
    n : int
        Total number of samples in the signal (len(signal)).
    fs : float
        Sampling frequency (Hz).
    min_len_sec : float, optional
        Minimum duration (in seconds) for a segment to keep.
        Shorter segments are dropped.
    pad_sec : float, optional
        Extra padding to add before and after each segment (seconds).

    Returns
    -------
    merged : list of tuples
        Cleaned list of (start_index, end_index) segments after
        padding, merging, and removing very short events.
    """

    if not segs:
        return []

    pad = int(round(pad_sec * fs))
    min_len = int(round(min_len_sec * fs))

    padded_segments = []
    for start, end in segs:
        start = max(0, start - pad)
        end = min(n - 1, end + pad)
        padded_segments.append((start, end))

    padded_segments.sort()

    merged = [padded_segments[0]]
    for start, end in padded_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    cleaned_segments = []
    for start, end in merged:
        if (end - start + 1) >= min_len:
            cleaned_segments.append((start, end))

    return cleaned_segments

def repair_motion_linear(signal, segs, fs, plot=True):
    """
    Repair motion artifact segments using simple linear interpolation.

    Parameters
    ----------
    signal : array-like
        The raw O2Hb or HHb signal (1D time series).
    segs : list of tuples
        List of (start_index, end_index) segments indicating motion artifact windows.
    fs : float
        Sampling frequency in Hz.
    plot : bool, optional
        If True, plot the original and repaired signals for visual comparison.

    Returns
    -------
    repaired_signal : ndarray
        The signal after linear interpolation over motion segments.

    Notes
    -----
    - Each motion segment (start:end) is replaced by a straight line
      connecting the signal value immediately before and after the segment.
    - If a segment touches the start or end of the recording, that edge
      is left unchanged.
    """
    signal = np.array(signal)
    n = len(signal)

    repaired = signal.copy()

    good = np.ones(n, dtype=bool)
    for start, end in segs:
        good[start:end + 1] = False

    idx = np.arange(n)
    repaired[~good] = np.interp(idx[~good], idx[good], signal[good])

    if plot:
        t = np.arange(n) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, color='black', alpha=0.7, label='Original signal')
        plt.plot(t, repaired, color='dodgerblue', linewidth=1.5, label='Repaired (linear interpolation)')

        # highlight motion segments
        for start, end in segs:
            plt.axvspan(t[start], t[end], color='tomato', alpha=0.3)

        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title('Motion Artifact Repair (Linear Interpolation)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return repaired

def repair_motion_scholkmann(signal, segs, fs, smoothness=0, pre_time=0.5, post_time=0.5, plot=False):
    """
    Repair motion artifacts using the cubic spline approach described by Scholkmann et al. (2010).

    Parameters
    ----------
    signal : array-like
        The raw O2Hb or HHb signal (1D time series).
    segs : list of tuples
        List of (start_index, end_index) segments indicating motion artifact windows.
    fs : float
        Sampling frequency in Hz.
    smoothness : float, optional
        Smoothing factor for UnivariateSpline (default=0 for an exact fit).
    pre_points : int, optional
        Number of clean samples to use before each motion segment as anchor points.
    post_points : int, optional
        Number of clean samples to use after each motion segment as anchor points.
    plot : bool, optional
        If True, plot the original and repaired signals.

    Returns
    -------
    repaired_signal : ndarray
        The signal after cubic spline repair and re-leveling.
    """

    signal = np.array(signal)
    n = len(signal)
    repaired = signal.copy()

    pre_points = int(round(pre_time * fs))
    post_points = int(round(post_time * fs))

    # --- Loop through each motion segment ---
    for start, end in segs:
        # define safe boundaries for anchor points
        pre_start = max(0, start - pre_points)
        post_end = min(n - 1, end + post_points)

        # indices before and after the artifact (used for spline fitting)
        anchor_idx = np.concatenate((
            np.arange(pre_start, start),
            np.arange(end + 1, post_end + 1)
        ))

        # if too few points, skip this segment
        if len(anchor_idx) < 4:
            continue

        # fit a cubic spline to the anchor points
        spline = UnivariateSpline(anchor_idx, signal[anchor_idx], s=smoothness)

        # replace motion-contaminated samples with spline values
        seg_idx = np.arange(start, end + 1)
        repaired[seg_idx] = spline(seg_idx)

        # --- re-level correction (continuity adjustment) ---
        # ensure smooth connection at the right boundary
        if end + 1 < n:
            offset = signal[end + 1] - repaired[end]
            repaired[end + 1:] += offset

    # --- Optional plot ---
    if plot:
        t = np.arange(n) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, color='black', alpha=0.7, label='Original signal')
        plt.plot(t, repaired, color='green', linewidth=1.5, label='Repaired (Scholkmann spline)')

        # highlight motion segments
        for start, end in segs:
            plt.axvspan(t[start], t[end], color='tomato', alpha=0.3)

        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title('Motion Artifact Repair (Scholkmann 2010 - Cubic Spline)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return repaired

def butter_bandpass_filtfilt(x, fs, low=0.01, high=0.30, order=4, plot=False):
    """
    Zero-phase Butterworth band-pass for fNIRS.

    Parameters
    ----------
    x : 1D array
    fs : float         # sampling rate (Hz)
    low, high : float  # cutoffs in Hz (e.g., 0.01–0.30 for HRF; 0.07–0.14 for MW)
    order : int        # 2–4 is typical; higher = steeper but more ringing

    Returns
    -------
    y : 1D array  # filtered signal (same length)
    """
    x = np.array(x)

    nyq = fs / 2.0
    wn = [low/nyq, high/nyq]
    if not (0 < wn[0] < wn[1] < 1):
        raise ValueError("Cutoffs must satisfy 0 < low < high < fs/2.")

    b, a = butter(order, wn, btype='band')
    # keep padlen safe for short records
    padlen = min(len(x)-1, 3*max(len(a), len(b)))
    y = filtfilt(b, a, x, padtype='odd', padlen=padlen)
    if plot:
        t = np.arange(len(x)) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, x, label='Original signal', color='black', alpha=0.6)
        plt.plot(t, y, label='Filtered signal', color='royalblue', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title(f'Butterworth Band-pass {low}-{high} Hz (order={order})')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return y

def butter_bandpass_filtfilt_SOS(x, fs, low=0.01, high=0.30, order=4, plot=False, demean=False):
    """
    Zero-phase Butterworth band-pass for fNIRS (stable SOS implementation).

    Parameters
    ----------
    x : 1D array
    fs : float
    low, high : float   # Hz (e.g., 0.01–0.30 for HRF; 0.07–0.14 for MW)
    order : int         # 2–4 typical; higher = steeper & more ringing
    plot : bool
    demean : bool       # if True, remove mean before filtering (optional)

    Returns
    -------
    y : 1D array
    """
    x = np.asarray(x, dtype=float).copy()

    # Handle NaNs so filtering doesn't fail
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        x[~good] = np.interp(idx[~good], idx[good], x[good])

    if demean:
        x -= x.mean()

    # Design Butterworth in numerically-stable SOS form
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')

    # Zero-phase forward/backward filtering
    y = sosfiltfilt(sos, x)

    if plot:
        t = np.arange(len(x)) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, x, label='Original', color='black', alpha=0.6)
        plt.plot(t, y, label=f'Band-pass {low}-{high} Hz', color='royalblue', lw=1.5)
        plt.xlabel('Time (s)'); plt.ylabel('Signal')
        plt.title(f'Butterworth (SOS) order={order}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return y

def Principal_component_analysis(list_of_signals, plot=False):

    shape = len(list_of_signals)
    z_signals_list = []
    for signal in list_of_signals:
        z_signal = z_transform(signal, 1, 0)
        z_signals_list.append(z_signal)

    X = np.column_stack(z_signals_list)

    pca = PCA()                              # keep all PCs
    PCs = pca.fit_transform(X)               # (N_samples, N_signals)
    pcs_list = [PCs[:, i] for i in range(PCs.shape[1])]
    explained_var = pca.explained_variance_ratio_
    print(explained_var)
    print(pcs_list)

    if plot:
        t = np.arange(X.shape[0])
        fig, ax = plt.subplots(figsize=(10, 4))
        # plot inputs with vertical offsets for clarity
        for i in range(len(z_signals_list)):
            ax.plot(t, z_signals_list[i], label=f'Signal {i + 1}')
        ax.plot(t, pcs_list[0], color='black', lw=2, label='PC1')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_title('Input short-channel signals and PC1')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

    return pcs_list, explained_var

def RMS(original, filtered):
    residual = original - filtered
    rms = np.sqrt(np.mean(residual ** 2))
    return rms

import math
import scipy.stats
from scipy import signal
import matplotlib.pyplot as plt
import statistics
import numpy as np
from numpy.fft import fft, fftfreq
import colorednoise as cn
import os

def FFT(var,fs):
    dt = 1 / fs
    freqs = fftfreq(len(var), dt)
    mask = freqs > 0
    Y = fft(var)
    pSpec = 2 * ((abs(Y) / len(var)) ** 2)

    f = freqs[mask]
    a = pSpec[mask]
    plt.plot(f,a)
    plt.show()
    sumA=[0]
    for i in range(1,len(a)):
        sumA.append(a[i]+sumA[i-1])

    prec90= []
    prec95 = []
    prec99 = []
    percA = [(sumA[i] / sumA[-1])*100 for i in range(len(sumA))]
    for p in percA:
        prec90.append(abs(p - 90))
        prec95.append(abs(p - 95))
        prec99.append(abs(p - 99))
    for i in range(len(percA)):
        if prec90[i]==min(prec90):
            index90 = i
        if prec95[i]==min(prec95):
            index95 = i
        if prec99[i]==min(prec99):
            index99 = i
    return f[index90],f[index95],f[index99]

def q_to_ypr(q):
    if q:
        yaw = (math.atan2(2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[0] ** 2 + 2 * q[1] ** 2 - 1))
        roll = (-1 * math.asin(2 * q[1] * q[3] + 2 * q[0] * q[2]))
        pitch = (math.atan2(2 * q[2] * q[3] - 2 * q[0] * q[1], 2 * q[0] ** 2 + 2 * q[3] ** 2 - 1))
        return [yaw, pitch, roll]

def pyth2d(x1,y1,x2,y2):
    x=x2-x1
    y=y2-y1
    c=math.sqrt(x**2+y**2)
    return c

def compute_cop(X, Y, df_filtered):
    list_X_coordinates_left_plate = []
    list_Y_coordinates_left_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1L'][i] + df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i] + \
                df_filtered['CHANNEL_4L'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i])) / F_all
        list_X_coordinates_left_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3L'][i] + df_filtered['CHANNEL_4L'][i])) / F_all
        list_Y_coordinates_left_plate.append(y_coordinate)

    list_X_coordinates_right_plate = []
    list_Y_coordinates_right_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1R'][i] + df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i] + \
                df_filtered['CHANNEL_4R'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i])) / F_all
        list_X_coordinates_right_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3R'][i] + df_filtered['CHANNEL_4R'][i])) / F_all
        list_Y_coordinates_right_plate.append(y_coordinate)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    for i in range(len(list_X_coordinates_left_plate)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_left_plate[i] - X / 2)
        list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_left_plate[i] - Y / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_right_plate[i] - X / 2)
        list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_right_plate[i] - Y / 2)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms = []
    for i in range(len(list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[i] - X / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[i] + X / 2)

    list_X_coordinates_both_plates = []
    list_Y_coordinates_both_plates = []
    for i in range(len(list_X_coordinates_right_plate)):
        list_X_coordinates_both_plates.append((list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i] +
                                               list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i]) / 2)
        list_Y_coordinates_both_plates.append((list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[
                                                   i] +
                                               list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[
                                                   i]) / 2)
    return list_X_coordinates_both_plates,list_Y_coordinates_both_plates
    #return list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform

def peaks(var,distance,height):
    peaks, _ = signal.find_peaks(var, distance=distance, height=height)
    peaksAmp = [var[peak] for peak in peaks]

    return peaks,peaksAmp

def Linear_Interpolation(col, step, plus):
    n = len(col)
    newdf = []
    value = step
    while value < n - 1:
        if math.ceil(value) == math.floor(value):
            num = col[math.ceil(value) + plus]
        else:

            num = ((col[math.ceil(value) + plus] - col[math.floor(value) + plus]) * (value - math.floor(value))) / (
                    math.ceil(value) - math.floor(value)) + col[math.floor(value) + plus]

        newdf.append(num)
        value = value + step

    return newdf

def Butterworth(fs, fc, var):
    """ Parameter:
            fs:     sampling frequency
            fc:     cutoff frequency for example 30Hz
            var:    data series
    """


    b, a = signal.butter(N=2, Wn=fc, btype='low', fs=fs)
    return signal.filtfilt(b, a, var)

def Butterworth_highpass(fs,fc,var):
    """ Parameter:
            fs:     sampling frequency
            fc:     cutoff frequency for example 30Hz
            var:    data series
    """


    b, a = signal.butter(N=2, Wn=fc, btype='high', fs=fs)
    return signal.filtfilt(b, a, var)

def Butterworth_band(fs,fc,var):
    """ Parameter:
            fs:     sampling frequency
            fc:     last of lower and upper limit
            var:    data series
    """


    b, a = signal.butter(N=2, Wn=fc, btype='band', fs=fs)
    return signal.filtfilt(b, a, var)

def Average(lst):
    return sum(lst) / len(lst)

def Remove_drift(inte,index,cut):
    inte = inte[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]
    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_sl[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Remove_drift2(inte,inte2nd,index,cut):
    inte = inte[cut:]
    inte2nd = inte2nd[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]

    slope2, intercept, r, p, stderr = scipy.stats.linregress(index, inte2nd)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl2 = [i * slope2 for i in index]
    t_slope=[a-b for a,b in zip(t_sl,t_sl2)]

    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_slope[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Bland_Altman_plot(Var1,Var2,title):
    Difference = [v - m for v, m in zip(Var1, Var2)]
    Mean = [(v + m) / 2 for v, m in zip(Var1, Var2)]
    Bias = Average(Difference)
    StanDev = statistics.stdev(Difference)
    LowerLOA = Bias - 1.96 * StanDev
    UpperLOA = Bias + 1.96 * StanDev
    inlims=0
    uplim=0
    downlim=0
    for d in Difference:
        if d>=UpperLOA:
            uplim+=1
        elif d<=LowerLOA:
            downlim+=1
        else:
            inlims+=1

    print('Total points: ', len(Difference),
          '\nInside points number: ', inlims,
          '\nUp points number: ', uplim,
          '\nDown points number: ', downlim,
          '\nUp Perc: ', (uplim/len(Difference))*100,
          '\nDown Perc: ', (downlim/len(Difference))*100,
          '\nOut Perc: ', ((downlim + uplim) / len(Difference)) * 100)

    plt.show()
    plt.title('Bland Altman Plot {name}'.format(name=title), fontsize=16)
    plt.xlabel('Average', fontsize=16)
    plt.ylabel('Difference', fontsize=16)
    plt.scatter(Mean, Difference, color='grey', linewidths=1.5)
    plt.axhline(y=Bias, color='black')
    plt.axhline(y=LowerLOA, color='black', ls=':')
    plt.axhline(y=UpperLOA, color='black', ls=':')
    plt.show()

    #OutPerc = ((downlim + uplim) / len(Difference)) * 100
    res_list = [len(Difference), inlims, uplim, downlim, (uplim / len(Difference)) * 100,
                (downlim / len(Difference)) * 100, ((downlim + uplim) / len(Difference)) * 100]
    return res_list

def intergral(span,dt):
    rects = []
    for i in range(len(span) - 1):
        rects.append(((span[i] + span[i + 1]) * dt) / 2)
    integral = [rects[0]]
    for i in range(len(rects) - 1):
        integral.append(integral[i] + rects[i + 1])

    return integral

def derivative(array,fs):
    dt = 1/fs
    der = []

    array = list(array)

    for i in range(len(array)-1):
        der.append((array[i+1]-array[i])/dt)
    return der

def Pink_noise_generator():
    pass

def pink_noise_generator2(number_of_sets,targets_per_set,RM,time_per_set,percentage_of_mean,max_perc,min_perc,H=1):
    """
    Generation of pink noise signal
    Inputs:
            number_of_sets:     Total number of sets
            targets_per_set:    Total targets for each set
            RM:                 Max Force assessment
            time_per_set =      Total time of each set
            std:                Standard Deviation of generated time series
            H =                 Hurst exponent, the resulted signal will have Hurst exponent ± 0.02, default H = 1
            percentage_of_mean: This is the percentage of 1Rm which will be the mean value for our signal
     Outputs:
            signal:             A list with a pink noise signal
            Time:               A list with the Time
    """
    beta=1

    total_number_targets=number_of_sets*targets_per_set
    found_time_series = False
    i=0
    std = (10*RM)/100
    while not found_time_series:
        signal = cn.powerlaw_psd_gaussian(beta, total_number_targets)
        mean = (percentage_of_mean*RM)/100
        print("mean")
        print(mean)
        signal = [i + mean for i in signal]
        signal = [i * std for i in signal]
        mean_post = np.mean(signal)
        signal = [i + (mean - mean_post) for i in signal]
        DFA_a = DFA(signal)
        # the mean value is 70% of 1RM therefore we want our signal to be between 50% and 90% of 1RM
        max = (mean * max_perc) / 70
        min = (mean * min_perc) / 70
        i += 1
        if DFA_a > (H - 0.02) and DFA_a < (H + 0.02) and np.max(signal)<max and np.min(signal)>min:
            print(f"Found a pink signal with the right characteristics after {i} efforts")
            found_time_series = True
    total_time = number_of_sets * time_per_set
    step_for_time = total_time / (number_of_sets * targets_per_set)
    Time = np.arange(0, total_time, step_for_time)
    return signal,Time

def Residual_analysis(var, cutoff_frequencies, fs, number_of_fit_points=10, plot=True, save=None):

    """
    Perform residual analysis for one variable.

    Parameters
    ----------
    var : array-like
        Original time series.

    cutoff_frequencies : list or array-like
        Cutoff frequencies to test.

    fs : float, default=1000
        Sampling frequency in Hz.

    number_of_fit_points : int, default=10
        Number of highest cutoff frequencies used for the linear fit.

    plot : bool, default=True
        If True, display the residual-analysis plot.

    save : tuple or None, default=None
        Tuple containing (name, directory). For example:
        ("participant_1.png", r"C:\\Results\\Residual analysis")
        If None, the plot is not saved.

    Returns
    -------
    Fc : float
        Tested cutoff frequency whose RMS residual is closest to the
        y-intercept of the fitted line.
    """

    var = np.asarray(var, dtype=float)

    # Convert the cutoff frequencies to a sorted NumPy array
    cutoff_frequencies = np.asarray(
        sorted(cutoff_frequencies),
        dtype=float
    )

    # Calculate the RMS residual for every cutoff frequency
    rms_residual = []

    for fc in cutoff_frequencies:

        var_filtered = Butterworth(
            fs,
            fc,
            var
        )

        residual = var - var_filtered

        rms_residual.append(
            np.sqrt(np.mean(residual ** 2))
        )

    rms_residual = np.asarray(rms_residual)

    # Use the highest cutoff frequencies for the linear fit
    fit_frequencies = cutoff_frequencies[-number_of_fit_points:]
    fit_residuals = rms_residual[-number_of_fit_points:]

    # Fit the straight line: y = slope*x + intercept
    slope, intercept = np.polyfit(
        fit_frequencies,
        fit_residuals,
        1
    )

    # Evaluate the fitted line across all tested cutoff frequencies
    fitted_line = np.polyval(
        [slope, intercept],
        cutoff_frequencies
    )

    # Find the tested residual closest to the y-intercept
    closest_index = np.argmin(
        np.abs(rms_residual - intercept)
    )

    Fc = cutoff_frequencies[closest_index]

    # Create the figure when it needs to be displayed or saved
    if plot or save is not None:

        fig, ax = plt.subplots(figsize=(8, 5))

        # RMS residual curve
        ax.plot(
            cutoff_frequencies,
            rms_residual,
            marker='o',
            color='#2F7FBF',
            label='RMS residual'
        )

        # Points used for the linear fit
        ax.scatter(
            fit_frequencies,
            fit_residuals,
            color='#F28C38',
            s=55,
            zorder=3,
            label=f'Last {number_of_fit_points} cutoff values'
        )

        # Extrapolated fitted line
        ax.plot(
            cutoff_frequencies,
            fitted_line,
            linestyle='--',
            color='#F28C38',
            linewidth=2,
            label='Extrapolated linear fit'
        )

        # Horizontal line at the y-intercept
        ax.axhline(
            y=intercept,
            color='black',
            linestyle='--',
            label=f'Intercept = {intercept:.4f}'
        )

        # Selected cutoff frequency
        ax.axvline(
            x=Fc,
            color='#FF1515',
            linestyle=':',
            linewidth=2,
            label=f'Fc = {Fc:g} Hz'
        )

        # Mark the selected point
        ax.scatter(
            Fc,
            rms_residual[closest_index],
            color='#FF1515',
            s=70,
            zorder=4
        )

        ax.set_xlabel('Cutoff frequency (Hz)')
        ax.set_ylabel('RMS residual')
        ax.set_title('Residual Analysis')
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        # save must have the form: (name, directory)
        if save is not None:

            if not isinstance(save, tuple) or len(save) != 2:
                raise ValueError(
                    "save must be None or a tuple: (name, directory)."
                )

            name, save_dir = save

            # Create the directory if it does not already exist
            os.makedirs(save_dir, exist_ok=True)

            # Add .png if the name does not contain an extension
            if not os.path.splitext(name)[1]:
                name += '.png'

            save_path = os.path.join(
                save_dir,
                name
            )

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight'
            )

            print(f"Plot saved to: {save_path}")

        if plot:
            plt.show()
        else:
            plt.close(fig)

    return Fc

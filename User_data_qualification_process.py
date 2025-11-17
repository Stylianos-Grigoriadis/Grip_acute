import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip acute perturbation\Data\Data to screen\P6\Force data'
os.chdir(directory)

parts = directory.split(os.sep)
ID = parts[-2]
print(ID)

training_set_1 = pd.read_csv(r'Training_set_1.csv', skiprows=2)
training_set_2 = pd.read_csv(r'Training_set_2.csv', skiprows=2)
training_set_3 = pd.read_csv(r'Training_set_3.csv', skiprows=2)
training_set_4 = pd.read_csv(r'Training_set_4.csv', skiprows=2)
training_set_5 = pd.read_csv(r'Training_set_5.csv', skiprows=2)
training_set_6 = pd.read_csv(r'Training_set_6.csv', skiprows=2)
training_set_7 = pd.read_csv(r'Training_set_7.csv', skiprows=2)
training_set_8 = pd.read_csv(r'Training_set_8.csv', skiprows=2)
training_set_9 = pd.read_csv(r'Training_set_9.csv', skiprows=2)
training_set_with_pert = pd.read_csv(r'Training_set_with_pert.csv', skiprows=2)

print(training_set_1.columns)
training_set_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_1)
training_set_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_2)
training_set_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_3)
training_set_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_4)
training_set_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_5)
training_set_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_6)
training_set_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_7)
training_set_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_8)
training_set_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_9)
training_set_with_pert = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_with_pert)
print(training_set_1.columns)
print(training_set_1)

print(training_set_1['Performance'])
print(training_set_1['ClosestSampleTime'])


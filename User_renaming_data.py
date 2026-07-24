import os
import re
import shutil
from datetime import datetime

import pandas as pd


participants_information = pd.read_excel(
    r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals\Participants.xlsx'
)

raw_data_directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Raw Data'

directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen'

folder_to_move = ["Sine_4"]


def get_file_datetime(filename):
    pattern = re.compile(r"__(\d{2})([A-Za-z]{3})(\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$")
    match = pattern.search(filename)

    if match is None:
        return None

    day = int(match.group(1))
    month_text = match.group(2)
    year = 2000 + int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    second = int(match.group(6))

    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }

    month = months[month_text]

    return datetime(year, month, day, hour, minute, second)


def make_perturbation_names(perturbation_order, prefix):
    new_names = []

    up_counter = 1
    down_counter = 1

    for perturbation in perturbation_order:

        if "P_up" in perturbation:
            new_names.append(f"{prefix}_Pert_up_{up_counter}.csv")
            up_counter = up_counter + 1

        elif "P_down" in perturbation:
            new_names.append(f"{prefix}_Pert_down_{down_counter}.csv")
            down_counter = down_counter + 1

    return new_names


def make_brain_file_name(folder_name, old_filename):
    group_name, participant_number = folder_name.split("_", 1)

    group_letter = group_name[0].upper()
    file_extension = os.path.splitext(old_filename)[1]

    return f"Artinis_{group_letter}{participant_number}{file_extension}"


for folder_name in folder_to_move:

    print()
    print("Participant:", folder_name)

    old_participant_folder = os.path.join(raw_data_directory, folder_name)

    if not os.path.exists(old_participant_folder):
        print("Participant folder not found:", old_participant_folder)
        continue

    participant_row = participants_information[
        participants_information["ID"] == folder_name
    ]

    if participant_row.empty:
        print("No row found in Excel for:", folder_name)
        continue

    pre_perturbations = participant_row["Order of perturbations pre"].iloc[0]
    post_perturbations = participant_row["Order of perturbations post"].iloc[0]

    pre_perturbations = [x.strip() for x in pre_perturbations.split(",")]
    post_perturbations = [x.strip() for x in post_perturbations.split(",")]

    old_grip_data_folder = os.path.join(
        old_participant_folder,
        "Grip data"
    )

    old_brain_data_folder = os.path.join(
        old_participant_folder,
        "Brain data"
    )

    new_participant_folder = os.path.join(
        directory,
        folder_name
    )

    new_grip_data_folder = os.path.join(
        new_participant_folder,
        "Grip data"
    )

    new_brain_data_folder = os.path.join(
        new_participant_folder,
        "Brain data"
    )

    if not os.path.exists(old_grip_data_folder):
        print("Grip data folder not found")
        continue

    os.makedirs(new_grip_data_folder, exist_ok=True)

    grip_files = os.listdir(old_grip_data_folder)

    files_with_time = []

    for filename in grip_files:
        file_time = get_file_datetime(filename)

        if file_time is not None:
            files_with_time.append((filename, file_time))

    files_with_time.sort(key=lambda x: x[1])

    if len(files_with_time) != 24:
        print("Problem: I did not find 24 CSV files with time.")
        print("Number of files found:", len(files_with_time))
        continue

    print("\nTimes of all 24 files:")

    for i, (_, file_time) in enumerate(files_with_time, start=1):
        print(f"{i}. {file_time.strftime('%H:%M:%S')}")

    pre_files = files_with_time[0:6]
    training_files = files_with_time[6:16]
    post_files = files_with_time[16:22]
    isometric_files = files_with_time[22:24]

    pre_new_names = make_perturbation_names(
        pre_perturbations,
        "Pre"
    )

    post_new_names = make_perturbation_names(
        post_perturbations,
        "Post"
    )

    training_new_names = []

    for i in range(1, 11):
        training_new_names.append(f"Training_{i}.csv")

    isometric_new_names = [
        "Isometric_high.csv",
        "Isometric_low.csv"
    ]

    old_files = []
    new_names = []

    for i in range(6):
        old_files.append(pre_files[i][0])
        new_names.append(pre_new_names[i])

    for i in range(10):
        old_files.append(training_files[i][0])
        new_names.append(training_new_names[i])

    for i in range(6):
        old_files.append(post_files[i][0])
        new_names.append(post_new_names[i])

    for i in range(2):
        old_files.append(isometric_files[i][0])
        new_names.append(isometric_new_names[i])

    for old_name, new_name in zip(old_files, new_names):

        old_path = os.path.join(
            old_grip_data_folder,
            old_name
        )

        new_path = os.path.join(
            new_grip_data_folder,
            new_name
        )

        shutil.copy2(old_path, new_path)

        print(old_name, "->", new_name)

    if os.path.exists(old_brain_data_folder):

        brain_files = []

        for filename in os.listdir(old_brain_data_folder):
            file_path = os.path.join(
                old_brain_data_folder,
                filename
            )

            if os.path.isfile(file_path):
                brain_files.append(filename)

        if len(brain_files) == 1:

            os.makedirs(
                new_brain_data_folder,
                exist_ok=True
            )

            old_brain_filename = brain_files[0]

            new_brain_filename = make_brain_file_name(
                folder_name,
                old_brain_filename
            )

            old_brain_path = os.path.join(
                old_brain_data_folder,
                old_brain_filename
            )

            new_brain_path = os.path.join(
                new_brain_data_folder,
                new_brain_filename
            )

            shutil.copy2(
                old_brain_path,
                new_brain_path
            )

            print(
                old_brain_filename,
                "->",
                new_brain_filename
            )

        else:
            print(
                "Problem: Expected one brain data file, but found:",
                len(brain_files)
            )

    else:
        print("Brain data folder not found")

    print("Finished:", folder_name)

    print("\nTraining times, files 7 to 16:")

    for i in range(6, 16):
        file_time = files_with_time[i][1]
        print(file_time.strftime("%H:%M:%S"))
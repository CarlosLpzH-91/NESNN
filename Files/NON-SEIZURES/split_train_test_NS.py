import csv
import os
from shutil import copy2

with open('MARKS.csv', 'r') as file:
    csvreader = csv.reader(file)
    headers = next(csvreader)

    marks = [mark for mark in csvreader]

current_dir = os.getcwd()
count = 0
marks_train = []
marks_test = []
for element in marks:
    print(f'Doing Non-Seizure: {element[0]}')
    if count <= 35:
        folder = 'TEST'
        marks_test.append(element)
        count += 1
        print(count)
    else:
        folder = 'TRAIN'
        marks_train.append(element)

    current_file = os.path.join(current_dir, element[0])
    destination_dir = os.path.join(current_dir, folder)
    copy2(current_file, destination_dir)

# Saving CSV information of Train
with open('TRAIN/MARKS.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(marks_train)
# Saving CSV information of Test
with open('TEST/MARKS.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(marks_test)

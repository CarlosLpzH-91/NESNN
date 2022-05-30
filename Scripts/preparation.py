import csv

import scipy.io as scio
from numpy import resize as np_resize, zeros
import torch

from Manipulation.lhs import LHS
from Manipulation.de import DE
from Manipulation.kade import KADE
from Manipulation.bsa import BSA


def iterate(type1, type2, num_set, default_time=5000, encode=False):
    """
    Iterate over all MAT found in description files.

    :param encode:
    :param default_time:
    :param type1: Seizure or non-seizure (ALL CAPS)
    :param type2: Train or Test
    :param num_set: Number of set of testing
    :return: None
    """

    with open(f'../Files/{type1}/{type2}/MARKS.csv', 'r', newline='') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            eegFile = row[0]

            # If type1 doesn't have a start time, then timestep is constant
            # Else read the start time (Second element)
            if len(row) == 1:
                timestep = default_time
            else:
                timestep = int(row[1])

            trimmed_raw(eegFile, timestep, num_set, type1, type2)

            if encode:
                encoding(eegFile, num_set, type2)


def trimmed_raw(filename, timestamp, num_set, type1, type2):
    """
        Trimmed seizure file according to the seizure onset.
        This will help to shorten the files

        :param type1: Seizure or non-seizure (ALL CAPS)
        :param type2: Train or Test
        :param num_set: Number of the corresponding set
        :param timestamp: Time of seizure onset (5000 for NON-SEIZURES)
        :param filename: File to shorten
        :return: None
        """
    print(f' Trimming {filename} of {type1} - {type2} ...')
    data = scio.loadmat(f'../Files/{type1}/{type2}/{filename}')['data']
    data_trimmed = data[:, :timestamp]
    scio.savemat(f'../Files/Sets/Set{num_set}/{type2}/RAW/{filename}', {'data': data_trimmed})
    print(f' Finish Trimming')

    print('Writing information')
    # Label 0 for Non and Label 1 for Seizure
    label = 0 if type1 == 'NON-SEIZURES' else 1
    # If Non, then timestamp label goes to -1 (identifier)
    timestamp = -timestamp if type1 == 'NON-SEIZURES' else timestamp

    with open(f'../Files/Sets/Set{num_set}/LabelsGlobal.csv', 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([f'{filename[:-4]}', f'{label}', f'{timestamp}'])
    with open(f'../Files/Sets/Set{num_set}/{type2}/Labels.csv', 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([f'{filename[:-4]}', f'{label}', f'{timestamp}'])


def encoding(filename, num_set, type1, channels=18, decoded=False):
    """
    Trimmed seizure file according to the seizure onset.
    This will help to shorten the files

    :param channels:
    :param decoded:
    :param type1: Seizure or non-seizure (ALL CAPS)
    :param type2: Train or Test
    :param num_set: Number of the corresponding set
    :param timestamp: Time of seizure onset (5000 for NON-SEIZURES)
    :param filename: File to shorten
    :return: None
    """
    print(f' Encoding {filename} of {type1}')
    data = scio.loadmat(f'../Files/Sets/Set{num_set}/{type1}/RAW/{filename}')['data']
    dataEncoded = zeros(data.shape)
    dataDecoded = zeros(data.shape)
    values = []
    vectors = []

    # Converting raw eeg files into spikes
    shift = data.min()
    data = data - shift
    scale = data.max() * 2
    freq = 256
    ranges = [[0, 100],
              [0, 100],
              [0.8, 2.0]]
    size = 50
    cr = 2.43
    fx = 0.20
    n_update = 5
    prc_selection = 0.05
    for channel in range(channels):
        print(f'Channel {channel + 1}/{channels}')
        # Initial Sampling
        init_sample = LHS(numPoints=size,
                          rangeSize=ranges[0],
                          rangeCut=ranges[1],
                          rangeThreshold=ranges[2])
        # KADE 5%
        value, vector = KADE(samples=init_sample,
                             num_gen=50,
                             size=size,
                             cr=cr,
                             fx=fx,
                             ranges=ranges,
                             signal=data[channel],
                             scale=scale,
                             fs=freq,
                             tot_evals=0,
                             stop='Gen',
                             n_update=n_update,
                             prc_selection=prc_selection,
                             n_rest=20)

        print(value, vector)

        spiker = BSA(int(vector[0]), vector[1], vector[2], scale, freq)
        dataEncoded[channel] = spiker.encode(data[channel])
        values.append(value)
        vectors.append(vector)
        if decoded:
            dataDecoded[channel] = spiker.decode(dataEncoded[channel])

    scio.savemat(f'../Files/Sets/Set{num_set}/{type1}/Encoded/{filename}', {'data': dataEncoded})
    if decoded:
        scio.savemat(f'../Files/Sets/Set{num_set}/{type1}/Encoded/{filename}_Decoded', {'data': dataEncoded})
    print(f' Finish Encoding')

    print('Writing information')
    with open(f'../Files/Sets/Set{num_set}/{type1}/Encoded/Aptitudes.csv', 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([f'{filename[:-4]}', f'{vectors}', f'{values}'])


def save_torch(num_set, type_set, type_file, channels=18):
    names = []
    times = []

    data = []
    labels = []
    print(f' Reading labels from Set {num_set} - {type_set}')
    with open(f'../Files/Sets/Set{num_set}/{type_set}/Labels.csv', 'r', newline='') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for line in reader:
            names.append(line[0])
            labels.append(int(line[1]))

            # Since the marker -TIME in time means non seizure, adjust it
            t = int(line[2])
            if t < 0:
                t = -t

            times.append(t)

    timeMax = max(times)
    data = zeros((len(labels), timeMax, channels))
    for i, name in enumerate(names):
        print(f' Doing file {name} ({i + 1}/{len(labels)})')
        if type_file == 'encoded':
            rawData = scio.loadmat(f'../Files/Sets/Set{num_set}/{type_set}/Encoded/{name}')['data'].T
        else:
            rawData = scio.loadmat(f'../Files/Sets/Set{num_set}/{type_set}/RAW/{name}')['data'].T

        if rawData.shape[0] < timeMax:
            rawData = np_resize(rawData, (timeMax, rawData.shape[1]))

        data[i] = rawData
    if type_file == 'encoded':
        tensor_data = torch.tensor(data, dtype=torch.int8)
        torch.save(tensor_data, f'../Files/Sets/Set{num_set}/{type_set}/{type_set}_data_encoded.pt')
    else:
        tensor_data = torch.tensor(data, dtype=torch.float32)
        torch.save(tensor_data, f'../Files/Sets/Set{num_set}/{type_set}/{type_set}_data.pt')

    tensor_label = torch.tensor(labels, dtype=torch.int8)
    torch.save(tensor_label, f'../Files/Sets/Set{num_set}/{type_set}/{type_set}_labels.pt')

    tensor_times = torch.tensor(times, dtype=torch.int16)
    torch.save(tensor_times, f'../Files/Sets/Set{num_set}/{type_set}/{type_set}_times.pt')


# iterate('SEIZURES', 'TRAIN', 2, 1000, False)
# iterate('SEIZURES', 'TEST', 2, 1000, False)
# iterate('NON-SEIZURES', 'TRAIN', 2, 1000, False)
# iterate('NON-SEIZURES', 'TEST', 2, 1000, False)

save_torch('2', 'Test', 'encoded')
save_torch('2', 'Train', 'encoded')
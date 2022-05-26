"""
This script will take the filter values of aptitudes.csv to re-encode files

"""
import csv
import re
from Manipulation.bsa import BSA
import scipy.io as scio
from numpy import zeros

num_set = '2'
type1 = 'TRAIN'

with open(f'../Files/Sets/Set{num_set}/{type1}/Encoded/Aptitudes.csv', 'r', newline='') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for line in reader:
        # Name of file
        file = line[0]
        # Load file
        data = scio.loadmat(f'../Files/Sets/Set{num_set}/{type1}/RAW/{file}')['data']
        # Get parameters
        dataEncoded = zeros(data.shape)
        dataDecoded = zeros(data.shape)
        values = []
        vectors = []
        shift = data.min()
        data = data - shift
        scale = data.max() * 2
        freq = 256

        # To delete the first [array([
        values_str = line[1][8:]
        # Create array of all channels eliminating the ]), array([
        _values = values_str.split(']), array([')
        # Eliminate the last 4 characters of the last element
        _values[-1] = _values[-1][:-4]
        # Iterate over all values, eliminating the las 4 characters
        for i, str_val in enumerate(_values):
            # Create arrays of values via the ', ' separator
            values = re.split(', ', str_val)
            # Do encoding
            spiker = BSA(int(float(values[0])), float(values[1][1:]), float(values[2][1:]), scale, freq)
            dataEncoded[i] = spiker.encode(data[i])
            # Do decoding
            dataDecoded[i] = spiker.decode(dataEncoded[i])

        # Save encoded & decoded data
        scio.savemat(f'../Files/Sets/Set{num_set}/{type1}/Encoded/{file}', {'data': dataEncoded})
        scio.savemat(f'../Files/Sets/Set{num_set}/{type1}/Encoded/{file}_Decoded', {'data': dataEncoded})

import numpy as np
from scipy.stats import qmc


def LHS(numPoints, rangeSize=None, rangeCut=None, rangeThreshold=None, fs=256.0):
    """
    Latin Hypercube Sampling implementation for BSA parameters
    :param int numPoints: Number of points. Must be correct.
    :param list rangeSize: Range of Filter size.
    :param list rangeCut: Range of Cutoff frequency.
    :param list rangeThreshold: Range of Threshold.
    :param float fs: Sampling frequency of signal.
    :return: Array (Numpy) of sampling points (3, 2)
    """

    if rangeThreshold is None:
        rangeThreshold = [0.1, 2]
    if rangeCut is None:
        rangeCut = [0.001, fs / 2]
    if rangeSize is None:
        rangeSize = [1, 102]

    numVars = 3
    rangesVar = np.array([rangeSize,
                          rangeCut,
                          rangeThreshold])

    oddCounter = oddNumbers(rangeSize)
    if numPoints > oddCounter:
        print('LHS will have repeated values')
        # print(f'Invalid Number of points. Using {oddCounter} instead.')
        # numPoints = oddCounter

    # Sampling creation
    sampler = qmc.LatinHypercube(numVars, centered=True)
    samples = sampler.random(numPoints)

    # Scaling sampling
    scaleSamples = qmc.scale(samples, rangesVar[:, 0], rangesVar[:, 1])

    # Correcting sampling
    correction = [int(val) if int(val) % 2 else int(val) + 1 for val in scaleSamples[:, 0]]

    scaleSamples[:, 0] = correction

    return scaleSamples


def uniqueVals(listVar):
    seen = set()
    for x in listVar:
        if x in seen: return False
        seen.add(x)
    return True


def oddNumbers(range):
    oddTotal = (range[1] - range[0]) // 2

    if range[0] % 2 != 0 or range[1] % 2 != 0:
        oddTotal += 1

    return oddTotal

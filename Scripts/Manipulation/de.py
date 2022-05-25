import random
import numpy as np
from math import ceil
import copy
from Manipulation.bsa import get_real


def rand(bounds):
    """
    Rand method.

    :param list bounds: Boundaries [Low, High]
    :return: New bounded value
    """
    return bounds[0] + (random.random() * (bounds[1] - bounds[0]))


def corrections(vector, vectorRanges):
    """
    Corrections of vector parameters.

    :param list vector: Parameters vector [Size, Cutoff, Threshold]
    :param list vectorRanges: Parameters ranges [[Low_Size, High_Size],
                                                 [Low_Cutoff, High_Cutoff],
                                                 [Low_Threshold, High_Threshold]]
    :return: Corrected vector [List]
    """
    for ix, (value, v_range) in enumerate(zip(vector, vectorRanges)):
        if value < v_range[0] or value > v_range[1]:
            vector[ix] = rand(v_range)

    seiz_ceil = ceil(vector[0])
    vector[0] = seiz_ceil if seiz_ceil % 2 else seiz_ceil + 1

    return vector


class Individual:
    def __init__(self, model, vector, aptitude, sigma, ranges, source='R', maximization=True):
        """
        DE individual.

        :param Kriging or None model: Kriging model (If applied).
        :param np.ndarray vector: BSA parameters [Size, Cutoff, Threshold].
        :param float aptitude: Aptitude value.
        :param float sigma: Sigma value.
        :param list ranges: BSA parameters ranges [[Low_Size, High_Size],
                                                   [Low_Cutoff, High_Cutoff],
                                                   [Low_Threshold, High_Threshold]]
        :param str source: Source of values [Differential Evolution (DE) or Real value (KM)]
        :param bool maximization: Whether to maximize the objective function.
        """
        self.model = model
        self.vector = vector
        self.aptitude = aptitude
        self.sigma = sigma
        self.source = source
        self.ranges = ranges
        self.maximization = maximization

    def report(self):
        """
        Report parameters, aptitude and source.
        :return: parameters, aptitude and source [String].
        """
        return f'{self.vector} f(x) = {self.aptitude} - "{self.source}"'

    def give_values(self):
        """
        Give individual's values as dictionary.

        :return: Vector, aptitude, sigma and source [Dictionary].
        """
        return {'vector': self.vector,
                'aptitude': self.aptitude,
                'sigma': self.sigma,
                'source': self.source}

    def operator_model(self, cr, fx, v1, v2, v3, verbose=False):
        """
        Crossover-Mutation operator of KADE.

        :param float cr: CR mutation value.
        :param float fx: FX modification value.
        :param dict v1: Guest vector 1.
        :param dict v2: Guest vector 2.
        :param dict v3: Guest vector 3.
        :param bool verbose: Whether to show internal information.
        :return: None
        """
        # print(f'Current Kernel: {self.model.report_kernel()}')
        if verbose:
            print(f'Based: {self.report()}')
            print(f'V1: {v1["vector"]} - {v1["aptitude"]} - {v1["sigma"]} - {v1["source"]}')
            print(f'V2: {v2["vector"]} - {v2["aptitude"]} - {v2["sigma"]} - {v2["source"]}')
            print(f'V3: {v3["vector"]} - {v3["aptitude"]} - {v3["sigma"]} - {v3["source"]}')

        # Generate Jrand value
        j_rand = random.randint(0, self.vector.size - 1)
        if verbose:
            print(f'Jrand: {j_rand}')

        # Initialize trial
        trial = {'vector': np.zeros(self.vector.size),
                 'aptitude': 0,
                 'sigma': 0,
                 'source': 'K'}

        # Iterate over vector's features
        for j, (base, r1, r2, r3) in enumerate(zip(self.vector,
                                                   v1['vector'],
                                                   v2['vector'],
                                                   v3['vector'])):
            rand_v = random.random()
            if verbose:
                print(f'J: {j}')
                print(f'Rand: {rand_v}')

            # Verify conditions and form trial
            if rand_v < cr or j == j_rand:
                trial['vector'][j] = r1 + (fx * (r2 - r3))
            else:
                trial['vector'][j] = self.vector[j]

        # Correction trail vector
        trial['vector'] = corrections(trial['vector'], self.ranges)
        if verbose:
            print(f'Corrected: {trial["vector"]}')
        # Get the aptitude of trial
        prediction, sigma = self.model.predict([trial['vector']])
        trial['aptitude'] = prediction[0][0]
        trial['sigma'] = sigma[0]
        if verbose:
            print(f'Trial Evaluated: {trial}')

        # Compare target vs trial
        if self.maximization:
            if trial['aptitude'] > self.aptitude:
                self.vector = trial['vector']
                self.aptitude = trial['aptitude']
                self.sigma = trial['sigma']
                self.source = trial['source']

                if verbose:
                    print('Trial wins')
                    self.report()
        else:
            if trial['aptitude'] < self.aptitude:
                self.vector = trial['vector']
                self.aptitude = trial['aptitude']
                self.sigma = trial['sigma']
                self.source = trial['source']

                if verbose:
                    print('Trial wins')
                    self.report()

    def operator_BSA(self, cr, fx, v1, v2, v3, original_signal, signal_scale, frequency, verbose=False):
        """
        Crossover-Mutation operator of DE.

        :param float cr: CR mutation value.
        :param float fx: FX modification value.
        :param dict v1: Guest vector 1.
        :param dict v2: Guest vector 2.
        :param dict v3: Guest vector 3.
        :param np.ndarray original_signal: Original signal.
        :param float signal_scale: Signal scale.
        :param float frequency: Sampling frequency.
        :param bool verbose: Whether to show internal information.
        :return: None
        """
        if verbose:
            print(f'Based: {self.report()}')
            print(f'V1: {v1["vector"]} - {v1["aptitude"]} - {v1["sigma"]} - {v1["source"]}')
            print(f'V2: {v2["vector"]} - {v2["aptitude"]} - {v2["sigma"]} - {v2["source"]}')
            print(f'V3: {v3["vector"]} - {v3["aptitude"]} - {v3["sigma"]} - {v3["source"]}')

        # Generate Jrand value
        j_rand = random.randint(0, self.vector.size - 1)
        if verbose:
            print(f'Jrand: {j_rand}')

        # Initialize trial
        trial = {'vector': np.zeros(self.vector.size),
                 'aptitude': 0,
                 'sigma': 0,
                 'source': 'R'}

        # Iterate over vector's features
        for j, (base, r1, r2, r3) in enumerate(zip(self.vector,
                                                   v1['vector'],
                                                   v2['vector'],
                                                   v3['vector'])):
            rand_v = random.random()
            if verbose:
                print(f'J: {j}')
                print(f'Rand: {rand_v}')

            # Verify conditions and form trial
            if rand_v < cr or j == j_rand:
                trial['vector'][j] = r1 + (fx * (r2 - r3))
            else:
                trial['vector'][j] = self.vector[j]

        # Correction trail vector
        trial['vector'] = corrections(trial['vector'], self.ranges)
        if verbose:
            print(f'Corrected: {trial["vector"]}')
        # Get the aptitude of trial
        prediction = get_real(trial['vector'], original_signal, signal_scale, frequency)
        trial['aptitude'] = prediction
        if verbose:
            print(f'Trial Evaluated: {trial}')

        # Compare target vs trial
        if self.maximization:
            if trial['aptitude'] > self.aptitude:
                self.vector = trial['vector']
                self.aptitude = trial['aptitude']
                self.sigma = trial['sigma']
                self.source = trial['source']

                if verbose:
                    print('Trial wins')
                    self.report()
        else:
            if trial['aptitude'] < self.aptitude:
                self.vector = trial['vector']
                self.aptitude = trial['aptitude']
                self.sigma = trial['sigma']
                self.source = trial['source']

                if verbose:
                    print('Trial wins')
                    self.report()


def stop_condition(method, current_gen, num_gen, current_evals, num_evals):
    if method == 'Gen':
        return current_gen != num_gen
    elif method == 'Eval':
        return current_evals <= num_evals
    elif method == 'Hyb':
        return current_gen != num_gen and current_evals <= num_evals


def DE(samples, num_gen, size, cr, fx, ranges, signal, scale, fs, total_eval, stop='Gen', maximization=True):
    """
    Pure DE/rand/1/bin.

    :param np.ndarray samples: Initial samples.
    :param int num_gen: Number of generations.
    :param int size: Population size.
    :param float cr: CR mutation value.
    :param float fx: FX modification value.
    :param list ranges: Ranges of variables [[Low_Size, High_Size],
                                             [Low_Cutoff, High_Cutoff],
                                             [Low_Threshold, High_Threshold]].
    :param np.ndarray signal: Signal to be transform.
    :param float scale: Signal scale.
    :param int fs: Sampling frequency.
    :param int total_eval: Total number of evaluations.
    :param str stop: Stop condition. Acceptable values are:
                                        'Gen'   -> Stop at num_gen generation.
                                        'Eval'  -> Stop at total_eval evaluation.
                                        'Hyb'   -> Stop at num_gen generation or total_eval evaluation
                                                    (whichever happens first).
    :param bool maximization: Whether the objective function must be maximize.
    :return: bests: All best individual per generation [List of class Individual].
             best_aptitude: Best last aptitude [Float].
             population: Last population [List of class Individual].
    """
    # Variables
    current_gen = 0
    evals = size
    bests = []

    # Initial population
    aptitudes = np.array([get_real(sample, signal, scale, fs) for sample in samples])
    population = [Individual(None, vector, aptitude, 0, ranges, source='R')
                  for vector, aptitude in zip(samples, aptitudes)]

    if maximization:
        population.sort(key=lambda x: x.aptitude, reverse=True)
    else:
        population.sort(key=lambda x: x.aptitude)

    # Initial best
    bests.append(copy.deepcopy(population[0]))
    print(f'Gen 0: {population[0].report()}')

    # DE: Main Loop
    while stop_condition(stop, current_gen, num_gen, evals, total_eval):
        # Increase generation counter
        current_gen += 1
        evals += size
        # DE: Mutation-Crossover
        all_targets = [target.give_values() for target in population]
        for i in range(size):
            # Possible guest vectors
            candidates = [candidate for j, candidate in enumerate(all_targets) if j != i]
            # Select guest vectors
            choices = random.sample(candidates, 3)
            # Do operation
            population[i].operator_BSA(cr, fx, choices[0], choices[1], choices[2], signal, scale, fs)

        if maximization:
            population.sort(key=lambda x: x.aptitude, reverse=True)
        else:
            population.sort(key=lambda x: x.aptitude)

        # Best
        bests.append(copy.deepcopy(population[0]))
        print(f'Gen {current_gen}: {population[0].report()}')

    # Get best values
    best_aptitude = [best.aptitude for best in bests]

    return bests, best_aptitude, population, population[0].vector
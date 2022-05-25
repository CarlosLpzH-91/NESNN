import random
from scipy.stats import norm
import numpy as np
from math import ceil
import copy
import matplotlib.pyplot as plt
from Manipulation.bsa import get_real
from Manipulation.de import Individual
from Manipulation.kriging import Kriging


def EI(y, std, y_opt=0.0, xi=0.01):
    """
    Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    ----------- Modified version of SciKit-Optimize:
    https://scikit-optimize.github.io/stable/_modules/skopt/acquisition.html#gaussian_ei

    Parameters
    ----------
    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.

    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    values : array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
        :param xi:
        :param y_opt:
        :param std:
        :param y:
    """
    # -------------------------- Modification
    mu = y.reshape(1, -1)[0] * -1

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    return values


def POI(y, std, y_opt):
    mu = y.reshape(1, -1)[0] * -1

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    values[mask] = cdf

    return values


def LCB(y, std, w):
    mu = y.reshape(1, -1)[0] * -1
    values = mu - (w * std)
    return values


def selection(option, n_selection, y, sigma, y_best=0.0, xi=1.0, w=2):
    criteria = []
    # Uncertainty
    if option == 'U':
        criteria = sigma
    elif option == 'E':
        criteria = EI(y, sigma, -y_best, xi)
    elif option == 'P':
        criteria = POI(y, sigma, -y_best)
    elif option == 'L':
        criteria = LCB(y, sigma, w)
    else:
        print(f'{option} is not a valid option')

    # Selected indices
    return criteria.argsort()[-n_selection:]


def test_one_variable(model, original, signal_scale, frequency, lml, variable='Cutoff', num=0):
    """
    Test prediction on single variables.

    :param Kriging model: Kriging model.
    :param np.ndarray original: Original signal.
    :param float signal_scale: Signal scale.
    :param float frequency: Sampling frequency.
    :param float lml: Log-Marginal Likelihood of current model.
    :param str variable: Name of the variable to test.
    :param int num: Number of the iteration.
    :return: None
    """
    range_ = []
    k = 0
    if variable == 'Size':
        x = np.zeros((32, 3))
        range_ = np.arange(16, 80, 2) + 1
        x[:, 0] = range_
        x[:, 1] = 42
        x[:, 2] = 0.94
    elif variable == 'Cutoff':
        x = np.zeros((61, 3))
        range_ = np.linspace(20, 80, num=61)
        x[:, 0] = 57
        x[:, 1] = range_
        x[:, 2] = 0.94
        k = 1
    elif variable == 'Threshold':
        x = np.zeros((31, 3))
        range_ = np.arange(0.8, 1.1, 0.01)
        x[:, 0] = 57
        x[:, 1] = 42
        x[:, 2] = range_
        k = 2

    y_real = [get_real(x_, original, signal_scale, frequency) for x_ in x]
    y_pred, y_sigma = model.predict(x)

    plt.figure(f'{variable}-{num}')
    plt.title(f'Generation : {num}\nLML = {lml}')
    plt.plot(range_, y_real, 'k--', label='True')
    plt.plot(range_, y_pred, label='Prediction', color='green')
    plt.fill_between(np.ravel(x[:, k]),
                     np.ravel(y_pred - 1.96 * y_sigma.reshape(-1, 1)),
                     np.ravel(y_pred + 1.96 * y_sigma.reshape(-1, 1)),
                     color='lightgreen',
                     label='Confidence'
                     )
    plt.scatter(x[:, k], y_real, label='Observations', zorder=4)
    plt.legend()
    plt.ylabel('SNR')
    plt.xlabel(f'{variable}')
    plt.savefig(f'{variable}-{num}.png')
    plt.close()


def plot_comparison(name, real, predicted):
    """
    Ploting comparison of real and predicted values.

    :param str name: Name of the plot.
    :param real: Real values.
    :param predicted: Predicted values.
    :return: None
    """
    plt.figure(name)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    plt.scatter(real, predicted, label='Samples')
    plt.plot([real.min(), real.max()], [real.min(), real.max()],
             'k--', lw=3, label='Perfect prediction')

    plt.xlabel('True value (SNR)', fontsize=16)
    plt.ylabel('Predicted value (SNR)', fontsize=16)
    plt.legend()


def stop_condition(method, current_gen, num_gen, current_evals, num_evals):
    if method == 'Gen':
        return current_gen != num_gen
    elif method == 'Eval':
        return current_evals <= num_evals
    elif method == 'Hyb':
        return current_gen != num_gen and current_evals <= num_evals


def KADE(signal, scale, fs, ranges, samples, tot_evals,
         size, num_gen, cr, fx, n_update, prc_selection, select_option='U', stop='Gen',
         c_v=1.0, c_v_bounds=(1e-5, 1e5), rbf_ls=None, rbf_ls_bounds=None,
         n_rest=10, a=1e-6, norm_y=True, re_evaluate_km=False, maximization=True, verbose=False, plot_vars=False):
    """
    Kriging Assisted Differential Evolution implementation for BSA parameters optimization.

    :param np.ndarray signal: Signal to be transform.
    :param float scale: Signal scale.
    :param float fs: Sampling frequency.
    :param list ranges: Ranges of variables [[Low_Size, High_Size],
                                        [Low_Cutoff, High_Cutoff],
                                        [Low_Threshold, High_Threshold]].
    :param np.ndarray samples: Initial sampling.
    :param int tot_evals: Total number of evaluations.
    :param int size: Population size.
    :param int num_gen: Number of generations.
    :param float cr: CR mutation value.
    :param float fx: FX modification value.
    :param int n_update: Rate at wich the model will be updated.
    :param float prc_selection: Percentage of the population to be selected for model update.
    :param str select_option: Option to selection criteria. Valid options are:  'U' -> Uncertainty.
                                                                                'E' -> Expected Improvement.
                                                                                'P' -> Probability of Improvement.
                                                                                'L' -> Lower Confidence Bound
    :param str stop: Stop condition. Acceptable values are:
                                        'Gen'   -> Stop at num_gen generation.
                                        'Eval'  -> Stop at total_eval evaluation.
                                        'Hyb'   -> Stop at num_gen generation or total_eval evaluation
                                                    (whichever happens first).
    :param float c_v: Initial constant value for Kriging model.
    :param tuple c_v_bounds: Bounds of values for constant value of Kriging model (Low, High).
    :param np.ndarray rbf_ls: Initial values of Squared Exponential Kernel of Kriging Model.
    :param list rbf_ls_bounds: Bounds of values for Squared Exponential kernel of Kriging model [(Low, High),
                                                                                                 (Low, High).
                                                                                                 (Low, High)].
    :param int n_rest: Number of times the optimizer can reset.
    :param float a: Noise added at the diagonal.
    :param bool norm_y: Whether to normalize predicted values or not.
    :param bool re_evaluate_km: Whether to re-evaluate all population after updating the model.
    :param bool maximization: Whether the objective function must be maximize.
    :param bool verbose: Whether internal information should be shown.
    :param bool plot_vars: Whether create prediction plots of each individual variable at model update.
    :return: bests: All best individual per generation [List of class Individual].
             last_real_aptitudes: Aptitude values for all last population [List of Float].
             best_aptitude: Best last aptitude [Float].
             best_sigma: Best last sigma [Float].
             differences_aptitudes: All mean differences of aptitudes obtained from model updating [List of Float].
             sigmas:All mean sigmas obtained from model updating [List of Float].
             population: Last population [List of class Individual].
             lmls: Log-Marginal Likelihoods [List of Float].
    """
    # Variables
    current_gen = 0
    evals = size
    bests = []
    differences_aptitudes = []
    sigmas = []
    lmls = []
    y_best = 0.0

    if rbf_ls is None:
        rbf_ls = np.ones(3)
    if rbf_ls_bounds is None:
        rbf_ls_bounds = [(1e-5, 1e5)] * 3

    n_selection = ceil(size * prc_selection)

    # Initial Evaluations: Expensive
    aptitudes = np.array([get_real(sample, signal, scale, fs) for sample in samples])

    # Initial Kriging Model
    km = Kriging(c_v, c_v_bounds, rbf_ls, rbf_ls_bounds, n_rest, a, norm_y)
    km.train(samples, aptitudes)
    lml_num, lml_str = km.report_lml()
    lmls.append(lml_num)
    print(km.report_kernel())
    print(lml_str)

    if plot_vars:
        test_one_variable(km, signal, scale, fs, lml_num, 'Size', 0)
        test_one_variable(km, signal, scale, fs, lml_num, 'Cutoff', 0)
        test_one_variable(km, signal, scale, fs, lml_num, 'Threshold', 0)

    # Initial population
    population = [Individual(km, vector, aptitude, 0, ranges, source='R')
                  for vector, aptitude in zip(samples, aptitudes)]

    if maximization:
        population.sort(key=lambda x: x.aptitude, reverse=True)
    else:
        population.sort(key=lambda x: x.aptitude)

    # Initial best
    bests.append(copy.deepcopy(population[0]))
    print(f'Gen 0: {population[0].report()}')

    # DE: Main Loop
    while stop_condition(stop, current_gen, num_gen, evals, tot_evals):
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
            population[i].operator_model(cr, fx, choices[0], choices[1], choices[2])

        # KM: Update model:
        if not current_gen % n_update and current_gen != num_gen:
            # Criteria = Sigma
            # criteria = np.array([i.sigma for i in population])
            # Criteria = EI
            # For EI-PoI-LCB
            # y_best = population[0].aptitude
            y_best = 0.0
            if select_option == 'E' or select_option == 'P':
                y_best = [i.aptitude if i.source == 'R' else 0 for i in population][0]

            npApts = np.array([i.aptitude for i in population], dtype=np.int32)
            npSigm = np.array([i.sigma for i in population], np.int32)
            # criteria = EI(npApts, npSigm, -y_best, xi=1.0)
            # criteria = POI(npApts, npSigm, -y_best)
            #
            # # Select most uncertain points
            # max_indices = criteria.argsort()[-n_selection:]
            max_indices = selection(select_option, n_selection, npApts, npSigm, y_best)
            new_samples = np.array([population[i].vector for i in max_indices])

            selected_old_aptitudes = np.array([population[i].aptitude for i in max_indices])
            selected_sigmas = np.array([population[i].sigma for i in max_indices])

            # Expensive evaluation with new samples
            new_aptitudes = np.array([get_real(sample, signal, scale, fs) for sample in new_samples])

            # Update aptitudes of selected samples
            for i, update_aptitude in zip(max_indices, new_aptitudes):
                population[i].aptitude = update_aptitude
                population[i].sigma = 0
                population[i].source = 'R'

            # Append new samples and aptitudes
            samples = np.append(samples, new_samples, axis=0)
            aptitudes = np.append(aptitudes, new_aptitudes, axis=0)
            # r2_score = km.report_score(new_samples, new_aptitudes)
            if verbose:
                # print(f'R_Square: {r2_score}')
                print(f'New samples: {new_samples}')
                print(f'Old aptitudes: {selected_old_aptitudes}')
                print(f'New aptitudes: {new_aptitudes}')
                print(f'Selected sigmas: {selected_sigmas}')

            # Re-train model
            km.train(samples, aptitudes)
            lml_num, lml_str = km.report_lml()
            lmls.append(lml_num)
            print(km.report_kernel())
            print(lml_str)

            if plot_vars:
                test_one_variable(km, signal, scale, fs, lml_num, 'Size', current_gen)
                test_one_variable(km, signal, scale, fs, lml_num, 'Cutoff', current_gen)
                test_one_variable(km, signal, scale, fs, lml_num, 'Threshold', current_gen)

            if re_evaluate_km:
                # Re-evaluated population in KM
                for i, individual in enumerate(population):
                    if i not in max_indices:
                        prediction, sigma = km.predict([individual.vector])
                        individual.aptitude = prediction[0][0]
                        individual.sigma = sigma[0]
                        individual.source = 'K'
            # Mean values
            differences_aptitudes.append(np.mean(selected_old_aptitudes - new_aptitudes))
            sigmas.append(selected_sigmas.mean())

        if maximization:
            population.sort(key=lambda x: x.aptitude, reverse=True)
        else:
            population.sort(key=lambda x: x.aptitude)

        # Best
        bests.append(copy.deepcopy(population[0]))
        print(f'Gen {current_gen}: {population[0].report()}')

    # Re-Evaluate population in expensive
    last_real_aptitudes = [get_real(i.vector, signal, scale, fs) for i in population]
    for individual, real_value in zip(population, last_real_aptitudes):
        individual.aptitude = real_value
        individual.source = 'R'

    if maximization:
        population.sort(key=lambda x: x.aptitude, reverse=True)
    else:
        population.sort(key=lambda x: x.aptitude)

    # Get best values
    bests[-1] = copy.deepcopy(population[0])

    print(f'Last: {population[0].report()}')
    return population[0].aptitude, population[0].vector

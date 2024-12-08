from copy import deepcopy
from typing import Callable, OrderedDict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .sequential_sampling import sequential_sample_from_SEM_hat, sequential_sample_from_true_SEM

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def fit_gp(x, y, kernel=None, noise_variance=1e-5):
    """
    Fit a Gaussian Process using TFP.

    Parameters
    ----------
    x : np.ndarray
        Input data (features).
    y : np.ndarray
        Output data (targets).
    kernel : tfp.math.psd_kernels.PositiveSemidefiniteKernel, optional
        Kernel to use for the GP, by default ExponentiatedQuadratic.
    noise_variance : float, optional
        Noise variance for the GP, by default 1e-5.

    Returns
    -------
    tfp.distributions.GaussianProcessRegressionModel
        A fitted Gaussian Process regression model.
    """
    if kernel is None:
        kernel = tfk.ExponentiatedQuadratic()

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float64)

    gp = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        observation_index_points=x_tensor,
        observations=y_tensor,
        observation_noise_variance=noise_variance,
    )
    return gp


def update_sufficient_statistics_hat(
    temporal_index: int,
    target_variable: str,
    exploration_set: tuple,
    sem_hat: OrderedDict,
    node_parents: Callable,
    dynamic: bool,
    assigned_blanket: dict,
    mean_dict_store: dict,
    var_dict_store: dict,
    static_sem: dict,
    dynamic_sem: dict,
    timesteps: list,
    seed: int = 1,
) -> Tuple[Callable, Callable]:
    """
    Updates the mean and variance functions (priors) on our causal effects given the current exploration set.

    Parameters
    ----------
    temporal_index : int
        The current time index in the causal Bayesian network.
    target_variable : str
        The current target variable, e.g., "Y_1".
    exploration_set : tuple
        The current exploration set.
    sem_hat : OrderedDict
        Contains our estimated SEMs.
    node_parents : Callable
        Function that returns parents of the passed argument at the given time-slice.
    dynamic : bool
        Indicates if horizontal information is used.
    assigned_blanket : dict
        The assigned blanket thus far (i.e., up until the temporal index).
    mean_dict_store : dict
        Stores the updated mean function for this time index and exploration set.
    var_dict_store : dict
        Stores the updated variance function for this time index and exploration set.
    static_sem : dict
        Static SEM structure.
    dynamic_sem : dict
        Dynamic SEM structure.
    timesteps : list
        List of timesteps for the SEM.
    seed : int, optional
        The random seed, by default 1.

    Returns
    -------
    Tuple[Callable, Callable]
        Returns the updated mean and variance function.
    """
    if dynamic:
        intervention_blanket = deepcopy(assigned_blanket)
    else:
        intervention_blanket = deepcopy(assigned_blanket)
        assert all(intervention_blanket[key] is None for key in intervention_blanket.keys())

    def mean_function(x_vals):
        results = []
        for x in x_vals:
            if str(x) in mean_dict_store[temporal_index][exploration_set]:
                results.append(mean_dict_store[temporal_index][exploration_set][str(x)])
            else:
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx
                sample = sequential_sample_from_SEM_hat(
                    interventions=intervention_blanket,
                    sem_hat=sem_hat,
                    static_sem=static_sem,
                    dynamic_sem=dynamic_sem,
                    timesteps=timesteps,
                    node_parents=node_parents,
                )
                mean_val = sample[target_variable][temporal_index]
                mean_dict_store[temporal_index][exploration_set][str(x)] = mean_val
                results.append(mean_val)
        return np.array(results)

    def variance_function(x_vals):
        results = []
        for x in x_vals:
            if str(x) in var_dict_store[temporal_index][exploration_set]:
                results.append(var_dict_store[temporal_index][exploration_set][str(x)])
            else:
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx
                sample = sequential_sample_from_SEM_hat(
                    interventions=intervention_blanket,
                    sem_hat=sem_hat,
                    static_sem=static_sem,
                    dynamic_sem=dynamic_sem,
                    timesteps=timesteps,
                    node_parents=node_parents,
                    variance=True,
                )
                var_val = sample[target_variable][temporal_index]
                var_dict_store[temporal_index][exploration_set][str(x)] = var_val
                results.append(var_val)
        return np.array(results)

    return mean_function, variance_function


def update_sufficient_statistics(
    temporal_index: int,
    exploration_set: tuple,
    time_slice_children: dict,
    initial_sem: dict,
    sem: dict,
    dynamic: bool,
    assigned_blanket: dict,
) -> Tuple[Callable, Callable]:
    """
    Updates the sufficient statistics for static causal Bayesian optimization.

    Parameters
    ----------
    temporal_index : int
        The current time index.
    exploration_set : tuple
        The current exploration set.
    time_slice_children : dict
        Children nodes for the current time slice.
    initial_sem : dict
        The initial SEM.
    sem : dict
        The SEM.
    dynamic : bool
        Indicates if the SEM is dynamic.
    assigned_blanket : dict
        The intervention blanket.

    Returns
    -------
    Tuple[Callable, Callable]
        Updated mean and variance functions.
    """
    intervention_blanket = deepcopy(assigned_blanket)

    def mean_function(x_vals):
        results = []
        for x in x_vals:
            for intervention_variable, xx in zip(exploration_set, x):
                intervention_blanket[intervention_variable][temporal_index] = xx
            sample = sequential_sample_from_true_SEM(
                interventions=intervention_blanket, sem=sem, dynamic=dynamic
            )
            results.append(sample)
        return np.array(results)

    def variance_function(x_vals):
        return np.zeros((len(x_vals), 1))  # Variance assumed to be zero for deterministic SEM

    return mean_function, variance_function

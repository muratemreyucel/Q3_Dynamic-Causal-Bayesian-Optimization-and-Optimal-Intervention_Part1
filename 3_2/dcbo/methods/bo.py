import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.math import psd_kernels as tfk
from dcbo.bases.root import Root
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function
from dcbo.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange
from typing import Callable


class BO(Root):
    def __init__(
        self,
        G: str,
        sem: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        number_of_trials: int,
        base_target_variable: str,
        task: str = "min",
        exploration_sets: list = None,
        cost_type: int = 1,
        n_restart: int = 1,
        hp_i_prior: bool = True,
        debug_mode: bool = False,
        optimal_assigned_blankets: dict = None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "number_of_trials": number_of_trials,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)

        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        self.hp_i_prior = hp_i_prior
        self.seed = seed

        # Convert observational samples to dict of temporal lists
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

        # Initialize mean and variance functions
        self.mean_function = {t: {} for t in range(self.T)}
        self.variance_function = {t: {} for t in range(self.T)}

        for temporal_index in range(self.T):
            self.mean_function[temporal_index][self.exploration_sets[0]] = standard_mean_function
            self.variance_function[temporal_index][self.exploration_sets[0]] = zero_variance_adjustment

    def run(self):
        """
        Main optimization routine for Bayesian Optimization.
        """
        for temporal_index in trange(self.T, desc="Time index"):
            if self.debug_mode:
                print(f"Time: {temporal_index}")

            target = self.all_target_variables[temporal_index]
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index

            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):
                self._per_trial_computations(temporal_index, it, target, assigned_blanket)

            self._post_optimisation_assignments(target, temporal_index)

    def _get_assigned_blanket(self, temporal_index):
        """
        Retrieve the assigned blanket for a given temporal index.
        """
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        """
        Evaluate acquisition functions during optimization.
        """
        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None

            if self.seed_anchor_points is None:
                seed_to_pass = None
            else:
                seed_to_pass = int(self.seed_anchor_points * (temporal_index + 1) * it)

            (
                self.y_acquired[es],
                self.corresponding_x[es],
            ) = evaluate_acquisition_function(
                self.intervention_exploration_domain[es],
                bo_model,
                self.mean_function[temporal_index][es],
                self.variance_function[temporal_index][es],
                current_best_global_target,
                es,
                self.cost_functions,
                self.task,
                self.base_target_variable,
                dynamic=False,
                causal_prior=False,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                seed=seed_to_pass,
            )

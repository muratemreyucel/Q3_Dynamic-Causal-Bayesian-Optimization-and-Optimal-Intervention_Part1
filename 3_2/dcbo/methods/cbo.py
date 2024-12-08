from typing import Callable
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.math import psd_kernels as tfk
from dcbo.bases.root import Root
from dcbo.bayes_opt.causal_kernels import CausalRBF
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function
from dcbo.utils.sem_utils.sem_estimate import fit_arcs
from dcbo.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange


class CBO(Root):
    def __init__(
        self,
        G: str,
        sem: Callable,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        exploration_sets: list,
        number_of_trials: int,
        base_target_variable: str,
        ground_truth: list = None,
        estimate_sem: bool = True,
        task: str = "min",
        n_restart: int = 1,
        cost_type: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        concat: bool = False,
        optimal_assigned_blankets: dict = None,
        n_obs_t: int = None,
        hp_i_prior: bool = True,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "make_sem_estimator": make_sem_estimator,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "estimate_sem": estimate_sem,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "use_mc": use_mc,
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)

        self.concat = concat
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.n_obs_t = n_obs_t
        self.hp_i_prior = hp_i_prior
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points

        # Fit Gaussian processes to emissions
        self.sem_emit_fncs = fit_arcs(self.G, self.observational_samples, emissions=True)

        # Convert observational samples to dict of temporal lists
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run(self):
        """
        Main optimization routine for Causal Bayesian Optimization.
        """
        if self.debug_mode:
            assert self.ground_truth is not None, "Provide ground truth to plot surrogate models"

        for temporal_index in trange(self.T, desc="Time index"):
            target = self.all_target_variables[temporal_index]
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            self._update_observational_data(temporal_index=temporal_index)
            self._update_interventional_data(temporal_index=temporal_index)

            if temporal_index > 0 and (self.online or isinstance(self.n_obs_t, list)):
                self._update_sem_emit_fncs(temporal_index)

            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):
                if it == 0:
                    self.trial_type[temporal_index].append("o")
                    sem_hat = self.make_sem_hat(G=self.G, emission_fncs=self.sem_emit_fncs)

                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=False,
                        assigned_blanket=self.empty_intervention_blanket,
                        updated_sem=sem_hat,
                    )
                    self._update_opt_params(it, temporal_index, best_es)
                else:
                    if self.trial_type[temporal_index][-1] == "o":
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)

                    self._per_trial_computations(temporal_index, it, target, assigned_blanket)

            self._post_optimisation_assignments(target, temporal_index)

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
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self.mean_function[temporal_index][es] = standard_mean_function
                    self.variance_function[temporal_index][es] = zero_variance_adjustment

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
                dynamic=True,
                causal_prior=True,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                seed=seed_to_pass,
            )

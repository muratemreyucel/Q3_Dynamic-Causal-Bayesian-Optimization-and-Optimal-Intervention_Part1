from typing import Callable
from tqdm import trange
from dcbo.bases.dcbo_base import BaseClassDCBO
from dcbo.utils.utilities import convert_to_dict_of_temporal_lists
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function


class DCBO(BaseClassDCBO):
    def __init__(
        self,
        G: str,
        sem: Callable,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        exploration_sets: dict,
        number_of_trials: int,
        base_target_variable: str,
        task: str = "min",
        estimate_sem: bool = True,
        cost_type: int = 1,
        ground_truth: list = None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        optimal_assigned_blankets: dict = None,
        use_di: bool = False,
        transfer_hp_o: bool = False,
        transfer_hp_i: bool = False,
        hp_i_prior: bool = True,
        n_obs_t: int = None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        # Initialize the parent class
        super().__init__(
            G=G,
            sem=sem,
            make_sem_estimator=make_sem_estimator,
            observation_samples=observation_samples,
            intervention_domain=intervention_domain,
            intervention_samples=intervention_samples,
            exploration_sets=exploration_sets,
            estimate_sem=estimate_sem,
            base_target_variable=base_target_variable,
            task=task,
            cost_type=cost_type,
            number_of_trials=number_of_trials,
            ground_truth=ground_truth,
            n_restart=n_restart,
            use_mc=use_mc,
            debug_mode=debug_mode,
            online=online,
            num_anchor_points=num_anchor_points,
            args_sem=args_sem,
            manipulative_variables=manipulative_variables,
            change_points=change_points,
        )

        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.use_di = use_di
        self.transfer_hp_o = transfer_hp_o
        self.transfer_hp_i = transfer_hp_i
        self.hp_i_prior = hp_i_prior
        self.n_obs_t = n_obs_t
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points

        # Convert observational samples to dict of temporal lists
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run(self):
        """
        Main optimization routine for Dynamic Causal Bayesian Optimization.
        """
        for temporal_index in trange(self.T, desc="Time index"):
            target = self.all_target_variables[temporal_index]
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index

            self._update_observational_data(temporal_index)

            if self.use_di and temporal_index > 0:
                self._forward_propagation(temporal_index)

            if self.transfer_hp_o and temporal_index > 0:
                self._get_observational_hp_emissions(self.sem_emit_fncs, temporal_index)
                self._get_observational_hp_transition(self.sem_trans_fncs)

            if temporal_index > 0 and (self.use_di or self.online or isinstance(self.n_obs_t, list)):
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self._update_sem_fncs(temporal_index, temporal_index - 1)
                else:
                    self._update_sem_fncs(temporal_index)

            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):
                if it == 0:
                    self.trial_type[temporal_index].append("o")
                    sem_hat = self.make_sem_hat(
                        G=self.G, emission_fncs=self.sem_emit_fncs, transition_fncs=self.sem_trans_fncs
                    )

                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=True,
                        assigned_blanket=assigned_blanket,
                        updated_sem=sem_hat,
                    )

                    self._update_opt_params(it, temporal_index, self.best_initial_es)

                else:
                    if self.trial_type[temporal_index][-1] == "o":
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)

                    self._per_trial_computations(temporal_index, it, target, assigned_blanket)

            self._post_optimisation_assignments(target, temporal_index, DCBO=True)

    def _get_assigned_blanket(self, temporal_index):
        """
        Retrieve the assigned blanket for a given temporal index.
        """
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                return self.optimal_assigned_blankets[temporal_index]
            else:
                return self.assigned_blanket
        else:
            return self.assigned_blanket

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        """
        Evaluate acquisition functions for each exploration set.

        Parameters
        ----------
        temporal_index : int
            Current temporal index.
        current_best_global_target : float
            Current best value of the global target variable.
        it : int
            Current trial iteration.
        """
        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None

            # Evaluate acquisition function
            self.y_acquired[es], self.corresponding_x[es] = evaluate_acquisition_function(
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
                causal_prior=False,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
            )

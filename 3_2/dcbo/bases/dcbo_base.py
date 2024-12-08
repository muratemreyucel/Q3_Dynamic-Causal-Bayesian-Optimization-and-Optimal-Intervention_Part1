from copy import deepcopy
from typing import Callable

import numpy as np
from tensorflow_probability.python.math import psd_kernels as tfk
from tensorflow_probability import distributions as tfd
from dcbo.bayes_opt.causal_kernels import CausalRBF
from dcbo.utils.gp_utils import fit_gp
from dcbo.utils.sem_utils.sem_estimate import fit_arcs
from dcbo.bases.root import Root


class BaseClassDCBO(Root):
    def __init__(
        self,
        G: str,
        sem: Callable,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict = None,
        exploration_sets: list = None,
        estimate_sem: bool = False,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        number_of_trials: int = 10,
        ground_truth=None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
        optimal_assigned_blankets: dict = None,
        use_di: bool = False,
        transfer_hp_o: bool = False,
        transfer_hp_i: bool = False,
        hp_i_prior: bool = True,
        n_obs_t: int = None,
    ):
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

        # Fit Gaussian Processes to emission and transition arcs
        self.sem_emit_fncs = fit_arcs(self.G, self.observational_samples, emissions=True)
        self.sem_trans_fncs = fit_arcs(self.G, self.observational_samples, emissions=False)

    def _update_sem_fncs(self, temporal_index: int, temporal_index_data: int = None):
        """
        Update SEM functions for online inference.

        Parameters
        ----------
        temporal_index : int
            Current temporal index.
        temporal_index_data : int, optional
            Data temporal index, by default None.
        """
        if temporal_index_data is None:
            temporal_index_data = temporal_index

        self._update_sem_emit_fncs(temporal_index, t_index_data=temporal_index_data)
        self._update_sem_trans_fncs(temporal_index, t_index_data=temporal_index_data)

    def _update_sem_emit_fncs(self, t: int, t_index_data: int = None):
        for pa in self.sem_emit_fncs[t]:
            xx, yy = self._get_sem_emit_obs(t, pa, t_index_data)
            if xx and yy:
                if t_index_data == t:
                    self.sem_emit_fncs[t][pa] = fit_gp(x=xx, y=yy)
                else:
                    temporal_index_pa = tuple(v.split("_")[0] + "_" + str(t_index_data) for v in pa)
                    assert temporal_index_pa in self.sem_emit_fncs[t_index_data], (
                        temporal_index_pa,
                        self.sem_emit_fncs[t_index_data].keys(),
                    )
                    self.sem_emit_fncs[t][pa] = self.sem_emit_fncs[t_index_data][temporal_index_pa]

    def _update_sem_trans_fncs(self, t: int, t_index_data: int = None):
        assert t > 0

        for y, pa in zip(self.sem, self.sem_trans_fncs[t]):
            xx = np.hstack([self.observational_samples[v.split("_")[0]][:, t - 1].reshape(-1, 1) for v in pa])
            yy = self.observational_samples[y][:, t].reshape(-1, 1)

            if t_index_data == t:
                self.sem_trans_fncs[pa] = fit_gp(x=xx, y=yy)
            else:
                temporal_index_data_pa = tuple(v.split("_")[0] + "_" + str(t_index_data - 1) for v in pa)
                assert temporal_index_data_pa in self.time_indexed_trans_fncs_pa[t_index_data], (
                    t,
                    t_index_data,
                    pa,
                    temporal_index_data_pa,
                    self.time_indexed_trans_fncs_pa[t_index_data],
                )
                self.sem_trans_fncs[pa] = self.sem_trans_fncs[temporal_index_data_pa]

    def _get_sem_emit_obs(self, t: int, pa: tuple, t_index_data: int):
        """
        Get observations for emission functions.

        Parameters
        ----------
        t : int
            Temporal index.
        pa : tuple
            Parent variables.
        t_index_data : int
            Temporal index for data.

        Returns
        -------
        tuple
            X (input) and Y (output) for SEM emission functions.
        """
        xx = np.hstack([self.observational_samples[v.split("_")[0]][:, t_index_data].reshape(-1, 1) for v in pa])
        yy = self.observational_samples[pa[-1].split("_")[0]][:, t_index_data].reshape(-1, 1)
        return xx, yy

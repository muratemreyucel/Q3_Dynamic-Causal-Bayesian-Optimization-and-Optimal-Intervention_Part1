from collections import OrderedDict
from copy import deepcopy
from typing import Callable

from networkx import MultiDiGraph
import numpy as np
from sklearn.neighbors import KernelDensity  # Still used for marginal KDE estimation
from dcbo.utils.dag_utils.adjacency_matrix_utils import get_emit_and_trans_adjacency_mats

from ..gp_utils import fit_gp
from ..utilities import select_sample


def fit_arcs(G: MultiDiGraph, data: dict, emissions: bool) -> dict:
    """
    Fit within (inter) time-slice arcs and between (intra) transitions functions connecting causal dynamic network.

    Parameters
    ----------
    G : MultiDiGraph
        Causal DAG.
    data: dict
        Observational samples from the true system.
    emissions: bool
        Whether to fit emissions or transitions.

    Returns
    -------
    dict
        Dictionary containing the estimated functions.
    """
    if emissions:
        # Emission adjacency matrix (doesn't contain entries for transition edges)
        A, _ = get_emit_and_trans_adjacency_mats(G)
    else:
        # Transition adjacency matrix
        _, A = get_emit_and_trans_adjacency_mats(G)

    edge_fit_track_mat = deepcopy(A)
    T = G.T
    nodes = np.array(G.nodes())
    fncs = {t: {} for t in range(T)}  # Estimated functions

    # Fit for nodes with multiple children (fork nodes)
    fork_idx = np.where(A.sum(axis=1) > 1)[0]
    fork_nodes = nodes[fork_idx]

    if any(fork_nodes):
        for i, v in zip(fork_idx, fork_nodes):
            coords = np.where(A[i, :] == 1)[0]
            ch = nodes[coords].tolist()  # Get children variable names
            var, t = v.split("_")
            t = int(t)
            xx = data[var][:, t].reshape(-1, 1)  # Independent regressor
            for j, y in enumerate(ch):
                # Estimand
                var_y, _ = y.split("_")
                yy = data[var_y][:, t].reshape(-1, 1)
                # Fit GP
                if v.split("_")[1] == y.split("_")[1]:
                    # Emissions
                    fncs[t][(v, j, y)] = fit_gp(x=xx, y=yy)
                else:
                    # Transitions
                    fncs[t + 1][(v, j, y)] = fit_gp(x=xx, y=yy)
                edge_fit_track_mat[i, coords[j]] -= 1

    # Fit marginal for source nodes
    if emissions:
        for v in nodes[np.where(A.sum(axis=0) == 0)]:
            var, t = v.split("_")
            t = int(t)
            xx = data[var][:, t].reshape(-1, 1)
            fncs[t][(None, v)] = KernelDensity(kernel="gaussian").fit(xx)

    # Fit remaining edges
    for i, j in zip(*np.where(edge_fit_track_mat == 1)):
        pa_y, t_pa = nodes[i].split("_")
        y, t_y = nodes[j].split("_")
        if emissions:
            assert t_pa == t_y, (i, j, nodes[i], nodes[j])
        else:
            assert t_pa != t_y, (i, j, nodes[i], nodes[j])
        t = int(t_y)
        xx = data[pa_y][:, t].reshape(-1, 1)
        yy = data[y][:, t].reshape(-1, 1)
        fncs[t][(nodes[i],)] = fit_gp(x=xx, y=yy)
        edge_fit_track_mat[i, j] -= 1

    assert edge_fit_track_mat.sum() == 0

    # Fit many-to-one estimates
    many_to_one = np.where(A.sum(axis=0) > 1)[0]
    if any(many_to_one):
        for i, v in zip(many_to_one, nodes[many_to_one]):
            y, y_t = v.split("_")
            t = int(y_t)
            pa_y = nodes[np.where(A[:, i] == 1)]
            assert len(pa_y) > 1, (pa_y, y, many_to_one)
            xx = np.hstack([data[vv.split("_")[0]][:, t].reshape(-1, 1) for vv in pa_y])
            yy = data[y][:, t].reshape(-1, 1)
            if y_t == pa_y[0].split("_")[1]:
                # Emissions
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)
            else:
                # Transitions
                assert t != 0, (t, pa_y, y)
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)

    return fncs


def build_sem_hat(G: MultiDiGraph, emission_fncs: dict, transition_fncs: dict = None) -> Callable:
    """
    Build SEM using fitted emission and transition functions.

    Parameters
    ----------
    G : MultiDiGraph
        Causal graphical model.
    emission_fncs : dict
        Dictionary of fitted emission functions.
    transition_fncs : dict
        Dictionary of fitted transition functions.

    Returns
    -------
    Callable
        SEM estimate from observational data, used in CBO and DCBO.
    """

    class SEMHat:
        def __init__(self):
            self.G = G
            nodes = G.nodes()
            n_t = len(nodes) / G.T
            assert n_t.is_integer()
            self.n_t = int(n_t)

        @staticmethod
        def _make_marginal() -> Callable:
            return lambda t, margin_id: emission_fncs[t][margin_id].sample()

        @staticmethod
        def _make_emit_fnc(moment: int) -> Callable:
            return lambda t, _, emit_input_vars, sample: emission_fncs[t][emit_input_vars].predict(
                select_sample(sample, emit_input_vars, t)
            )[moment]

        @staticmethod
        def _make_trans_fnc(moment: int) -> Callable:
            return lambda t, transfer_input_vars, _, sample: transition_fncs[t][transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        @staticmethod
        def _make_emit_plus_trans_fnc(moment: int) -> Callable:
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[t][transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        def static(self, moment: int) -> OrderedDict:
            assert moment in [0, 1]
            f = OrderedDict()
            for v in list(self.G.nodes)[: self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                else:
                    f[vv] = self._make_emit_fnc(moment)
            return f

        def dynamic(self, moment: int) -> OrderedDict:
            assert moment in [0, 1]
            f = OrderedDict()
            for v in list(self.G.nodes)[self.n_t : 2 * self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                elif all(int(vv.split("_")[1]) + 1 == int(v.split("_")[1]) for vv in G.predecessors(v)):
                    f[vv] = self._make_trans_fnc(moment)
                elif all(vv.split("_")[1] == v.split("_")[1] for vv in G.predecessors(v)):
                    f[vv] = self._make_emit_fnc(moment)
                else:
                    f[vv] = self._make_emit_plus_trans_fnc(moment)
            return f

    return SEMHat

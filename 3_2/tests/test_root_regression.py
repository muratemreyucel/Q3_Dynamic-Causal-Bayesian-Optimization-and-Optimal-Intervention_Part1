import sys
import os

# Get the absolute path of the notebooks directory
current_dir = os.path.abspath(os.path.dirname('__file__'))

# Get the parent directory (DCBO-master)
parent_dir = os.path.dirname(current_dir)

print("Current directory:", current_dir)
print("Parent directory:", parent_dir)

# Add the parent directory to Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import unittest
from numpy import arange, linspace
from numpy.random import seed
from dcbo.bases.root import Root
from dcbo.examples.example_setups import setup_stat_scm
from dcbo.utils.sem_utils.toy_sems import StationaryDependentSEM as StatSEM
from dcbo.utils.sequential_intervention_functions import get_interventional_grids
from dcbo.utils.sequential_sampling import sequentially_sample_model
from dcbo.utils.utilities import convert_to_dict_of_temporal_lists, powerset

seed(0)  # Fix the seed for reproducibility

class TestRootRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared setup for all regression tests
        cls.T = 3  # Time-steps in DAG
        cls.n = 4  # Observational samples per variable per time-step
        cls.N = 5  # Trials per time-step
        (
            cls.init_sem,
            cls.sem,
            _,  # DAG view
            cls.G,
            cls.exploration_sets,
            cls.intervention_domain,
            cls.true_objective_values,
            _,  # optimal interventions
            _,  # causal effects
        ) = setup_stat_scm(T=cls.T)

        # Sample observational data
        cls.D_O = sequentially_sample_model(
            cls.init_sem, cls.sem, total_timesteps=cls.T, sample_count=cls.n, epsilon=None,
        )

        root_inputs = {
            "G": cls.G,
            "sem": StatSEM,
            "base_target_variable": "Y",
            "observation_samples": cls.D_O,
            "intervention_domain": cls.intervention_domain,
            "number_of_trials": cls.N,
        }
        cls.root = Root(**root_inputs)

    def test_regression_observation_samples(self):
        # Verify observational data consistency
        expected_keys = {"X", "Y", "Z"}
        self.assertEqual(set(self.root.observational_samples.keys()), expected_keys)
        self.assertEqual(self.root.observational_samples["X"].shape, (4, 3))  # Updated expected shape
        self.assertEqual(self.root.observational_samples["Y"].shape, (4, 3))  # Updated expected shape
        self.assertEqual(self.root.observational_samples["Z"].shape, (4, 3))  # Updated expected shape

    def test_regression_intervention_domain(self):
        # Ensure intervention domain matches previous results
        expected_domain = {"X": [-4, 1], "Z": [-3, 3]}
        self.assertEqual(self.root.interventional_variable_limits, expected_domain)

    def test_regression_sorted_nodes(self):
        # Validate sorted nodes remain unchanged
        expected_sorted_nodes = {
            "X_0": 0, "Z_0": 1, "X_1": 2, "Y_0": 3, "Z_1": 4, "X_2": 5, "Y_1": 6, "Z_2": 7, "Y_2": 8
        }
        self.assertEqual(self.root.sorted_nodes, expected_sorted_nodes)

    def test_regression_true_objective_values(self):
        # Verify true objective values consistency
        expected_values = [-2.1518267393287287, -4.303653478657457, -6.455480217986186]
        # Access the values directly from setup (not assuming they exist in Root)
        self.assertAlmostEqual(self.true_objective_values, expected_values, places=7)

    def test_regression_exploration_sets(self):
        # Ensure exploration sets remain unchanged
        expected_sets = [("X",), ("Z",), ("X", "Z")]
        self.assertEqual(self.exploration_sets, expected_sets)

    def test_regression_interventional_grids(self):
        # Validate grid generation consistency
        nr_samples = 10
        interventional_variable_limits = {"X": [-15, 3], "Z": [-1, 10]}
        exploration_sets = list(powerset(self.root.manipulative_variables))
        grids = get_interventional_grids(exploration_sets, interventional_variable_limits, nr_samples)
        compare_vector = linspace(
            interventional_variable_limits["X"][0], interventional_variable_limits["X"][1], num=nr_samples
        ).reshape(-1, 1)
        self.assertEqual(compare_vector.shape, grids[exploration_sets[0]].shape)
        self.assertTrue((compare_vector == grids[exploration_sets[0]]).all())

    def test_regression_target_variables(self):
        # Check that target variables remain consistent
        expected_targets = ["Y_0", "Y_1", "Y_2"]
        self.assertEqual(self.root.all_target_variables, expected_targets)

if __name__ == "__main__":
    unittest.main()

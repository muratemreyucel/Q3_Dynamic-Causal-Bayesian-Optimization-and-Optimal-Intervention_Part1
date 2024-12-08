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
from numpy.random import seed
from dcbo.bases.root import Root
from dcbo.examples.example_setups import setup_stat_scm
from dcbo.utils.sem_utils.toy_sems import StationaryDependentSEM as StatSEM
from dcbo.utils.sequential_intervention_functions import get_interventional_grids
from dcbo.utils.sequential_sampling import sequentially_sample_model
from dcbo.utils.utilities import powerset

seed(0)  # Fix the seed for reproducibility

class TestRootIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Integration setup involving multiple components
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

        # Define Root class inputs
        root_inputs = {
            "G": cls.G,
            "sem": StatSEM,
            "base_target_variable": "Y",
            "observation_samples": cls.D_O,
            "intervention_domain": cls.intervention_domain,
            "number_of_trials": cls.N,
        }
        cls.root = Root(**root_inputs)

    def test_integration_workflow(self):
        # Verify that all components interact correctly in the Root workflow
        # Step 1: Observational data
        self.assertEqual(set(self.root.observational_samples.keys()), {"X", "Y", "Z"})
        self.assertEqual(self.root.observational_samples["X"].shape, (4, 3))  # Matches setup
        
        # Step 2: Check exploration sets
        self.assertEqual(self.root.exploration_sets, [("X", "Z")])  # Default assignment

        # Step 3: Generate interventional grids
        nr_samples = 10
        exploration_sets = list(powerset(self.root.manipulative_variables))
        grids = get_interventional_grids(exploration_sets, self.root.interventional_variable_limits, nr_samples)
        self.assertIn(("X",), grids)
        self.assertIn(("Z",), grids)

        # Step 4: Validate interventional variable limits
        self.assertEqual(self.root.interventional_variable_limits, {"X": [-4, 1], "Z": [-3, 3]})

        # Step 5: Ensure nodes are properly sorted
        self.assertEqual(
            self.root.sorted_nodes,
            {"X_0": 0, "Z_0": 1, "X_1": 2, "Y_0": 3, "Z_1": 4, "X_2": 5, "Y_1": 6, "Z_2": 7, "Y_2": 8},
        )

        # Step 6: Validate target variables
        self.assertEqual(self.root.all_target_variables, ["Y_0", "Y_1", "Y_2"])

        # Step 7: Confirm that interventional data is initially empty
        self.assertEqual(
            self.root.interventional_data_y, {t: {("X", "Z"): None} for t in range(self.T)}
        )
        self.assertEqual(
            self.root.interventional_data_x, {t: {("X", "Z"): None} for t in range(self.T)}
        )

if __name__ == "__main__":
    unittest.main()

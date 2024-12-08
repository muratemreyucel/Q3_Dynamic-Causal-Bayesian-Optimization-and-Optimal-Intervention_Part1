import unittest
import os
from exploration_set import (
    generate_causal_graph,
    get_exploration_set,
    save_graph_as_png,
)
from networkx.drawing.nx_agraph import write_dot


class TestRegression(unittest.TestCase):
    def setUp(self):
        """
        Set up resources needed for regression tests.
        """
        self.manipulable_nodes = ["M1", "M2", "M3"]
        self.expected_exploration_sets = [
            (),
            ("M1",),
            ("M2",),
            ("M3",),
            ("M1", "M2"),
            ("M1", "M3"),
            ("M2", "M3"),
        ]
        self.graph_T1_nodes = 15
        self.graph_T1_edges = 56
        self.test_png_file = "test_causal_graph.png"
        self.test_dot_file = "test_causal_graph.dot"

    def tearDown(self):
        """
        Clean up any resources created during tests.
        """
        if os.path.exists(self.test_png_file):
            os.remove(self.test_png_file)
        if os.path.exists(self.test_dot_file):
            os.remove(self.test_dot_file)

    def test_exploration_set_regression(self):
        """
        Ensure get_exploration_set produces consistent results.
        """
        result = get_exploration_set(self.manipulable_nodes, max_simultaneous_interventions=2)
        self.assertEqual(sorted(result), sorted(self.expected_exploration_sets))

    def test_causal_graph_regression(self):
        """
        Ensure generate_causal_graph creates consistent graph structure for T=1.
        """
        graph = generate_causal_graph(T=1)
        self.assertEqual(len(graph.nodes), self.graph_T1_nodes)
        self.assertEqual(len(graph.edges), self.graph_T1_edges)

    def test_save_graph_as_png_regression(self):
        """
        Ensure save_graph_as_png correctly saves a PNG file.
        """
        graph = generate_causal_graph(T=1)
        save_graph_as_png(graph, self.test_png_file)
        self.assertTrue(os.path.exists(self.test_png_file))


if __name__ == "__main__":
    unittest.main()

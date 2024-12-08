import unittest
import os
from exploration_set import (
    powerset,
    generate_causal_graph,
    get_exploration_set,
    save_graph_as_png,
)
from networkx.drawing.nx_agraph import write_dot
import networkx as nx


class TestExplorationSet(unittest.TestCase):
    def setUp(self):
        """
        Set up resources needed for the tests.
        """
        self.manipulable_nodes = ["M1", "M2", "M3"]
        self.test_dot_file = "test_causal_graph.dot"
        self.test_png_file = "test_causal_graph.png"

    def tearDown(self):
        """
        Clean up resources created during the tests.
        """
        if os.path.exists(self.test_dot_file):
            os.remove(self.test_dot_file)
        if os.path.exists(self.test_png_file):
            os.remove(self.test_png_file)

    def test_powerset(self):
        """
        Test the powerset function with various inputs.
        """
        # Test empty list
        self.assertEqual(powerset([]), [()])

        # Test single element
        self.assertEqual(powerset([1]), [(), (1,)])

        # Test multiple elements
        self.assertEqual(powerset([1, 2]), [(), (1,), (2,), (1, 2)])

        # Test larger set
        result = powerset([1, 2, 3])
        expected = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        self.assertEqual(result, expected)

    def test_generate_causal_graph(self):
        """
        Test the generate_causal_graph function for correct node and edge counts.
        """
        # Test with T=1
        graph = generate_causal_graph(T=1)
        self.assertEqual(len(graph.nodes), 15)  # 7 manipulable, 7 non-manipulable, 1 target
        self.assertEqual(len(graph.edges), 56)  # 49 topological + 7 target dependencies

        # Test with T=2
        graph = generate_causal_graph(T=2)
        self.assertEqual(len(graph.nodes), 30)  # 15 nodes * 2 time steps
        self.assertEqual(len(graph.edges), 127)  # 98 topological, 14 target, 15 temporal
        self.assertTrue(graph.has_edge("M1_0", "N1_0"))  # Dependency between manipulable and non-manipulable
        self.assertTrue(graph.has_edge("N1_0", "T_0"))  # Dependency to target
        self.assertTrue(graph.has_edge("M1_0", "M1_1"))  # Temporal dependency

    def test_get_exploration_set(self):
        """
        Test the get_exploration_set function for valid subsets.
        """
        # Test max_simultaneous_interventions = 2
        result = get_exploration_set(self.manipulable_nodes, max_simultaneous_interventions=2)
        self.assertIn(("M1", "M2"), result)
        self.assertIn(("M1",), result)
        self.assertIn(("M2", "M3"), result)
        self.assertNotIn(("M1", "M2", "M3"), result)  # Should not allow 3 interventions

        # Test empty manipulable nodes
        result = get_exploration_set([], max_simultaneous_interventions=2)
        self.assertEqual(result, [()])  # Only the empty set is valid

    def test_save_graph_as_png(self):
        """
        Test saving the graph as a PNG file.
        """
        graph = generate_causal_graph(T=1)
        save_graph_as_png(graph, self.test_png_file)
        self.assertTrue(os.path.exists(self.test_png_file))  # Ensure PNG file is created

    def test_save_graph_to_dot(self):
        """
        Test saving the graph to a DOT file.
        """
        graph = generate_causal_graph(T=1)
        write_dot(graph, self.test_dot_file)
        self.assertTrue(os.path.exists(self.test_dot_file))  # Ensure DOT file is created

    def test_large_graph(self):
        """
        Test generating a large causal graph to ensure scalability.
        """
        graph = generate_causal_graph(T=10)
        self.assertEqual(len(graph.nodes), 150)  # 15 nodes * 10 time steps
        self.assertTrue("M1_9" in graph.nodes)  # Ensure node exists
        self.assertTrue("M1_8" in graph.nodes)  # Ensure node exists
        self.assertTrue(graph.has_edge("M1_8", "M1_9"), "Temporal edge from M1_8 to M1_9 is missing")


if __name__ == "__main__":
    unittest.main()

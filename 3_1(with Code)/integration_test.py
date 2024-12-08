import unittest
import os
from exploration_set import (
    generate_causal_graph,
    get_exploration_set,
    save_graph_as_png,
    save_graph_to_dot,
    visualize_graph_from_dot,
)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """
        Set up resources needed for integration tests.
        """
        self.manipulable_nodes = [f"M{i}" for i in range(1, 8)]
        self.png_file = "integration_test_graph.png"
        self.dot_file = "integration_test_graph.dot"

    def tearDown(self):
        """
        Clean up resources created during the tests.
        """
        if os.path.exists(self.png_file):
            os.remove(self.png_file)
        if os.path.exists(self.dot_file):
            os.remove(self.dot_file)

    def test_full_workflow(self):
        """
        Test the complete workflow: generate graph, save as PNG and DOT, 
        retrieve exploration sets, and visualize graph.
        """
        try:
            # Generate causal graph
            graph = generate_causal_graph(T=3)
            self.assertGreater(len(graph.nodes), 0, "Graph nodes not generated.")
            self.assertGreater(len(graph.edges), 0, "Graph edges not generated.")

            # Save graph as PNG
            save_graph_as_png(graph, self.png_file)
            self.assertTrue(os.path.exists(self.png_file), "PNG file was not saved.")

            # Save graph as DOT
            save_graph_to_dot(graph, self.dot_file)
            self.assertTrue(os.path.exists(self.dot_file), "DOT file was not saved.")

            # Generate exploration sets
            exploration_sets = get_exploration_set(self.manipulable_nodes, max_simultaneous_interventions=3)
            self.assertIn(("M1", "M2"), exploration_sets, "Exploration set is missing expected values.")

            # Visualize graph (functionality test)
            visualize_graph_from_dot(self.dot_file)  # Visual test; no assertions

        except Exception as e:
            self.fail(f"Workflow test failed with exception: {e}")

    def test_large_graph_workflow(self):
        """
        Test workflow for a large graph with multiple time steps.
        """
        try:
            # Generate a large causal graph
            graph = generate_causal_graph(T=10)
            self.assertGreater(len(graph.nodes), 0, "Large graph nodes not generated.")
            self.assertGreater(len(graph.edges), 0, "Large graph edges not generated.")

            # Save graph as PNG and DOT
            save_graph_as_png(graph, self.png_file)
            save_graph_to_dot(graph, self.dot_file)
            self.assertTrue(os.path.exists(self.png_file), "Large graph PNG file was not saved.")
            self.assertTrue(os.path.exists(self.dot_file), "Large graph DOT file was not saved.")

        except Exception as e:
            self.fail(f"Large graph workflow test failed with exception: {e}")

    def test_invalid_graph_generation(self):
        """
        Test behavior with invalid parameters for graph generation.
        """
        with self.assertRaises(ValueError):
            generate_causal_graph(T=0)  # T must be >= 1


if __name__ == "__main__":
    unittest.main()

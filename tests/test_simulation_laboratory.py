"""Unit tests for the synthetic simulation and incremental laboratory."""
import unittest

from laboratory.engine import LaboratoryEngine
from simulation.config import SimulationConfig
from simulation.engine import SimulationEngine


class SimulationLaboratoryTests(unittest.TestCase):
    def test_generates_configured_users_and_events(self):
        simulation = SimulationEngine(SimulationConfig(
            user_count=5, interactions_per_user=2, random_seed=7
        ))
        events = simulation.run_batch(["Movie A", "Movie B", "Movie C"])
        self.assertEqual(len(simulation.users), 5)
        self.assertEqual(sum(event.event_type == "movie_rate" for event in events), 10)
        self.assertTrue(all(event.source == "simulation" for event in events))
        self.assertTrue(all(1 <= event.rating <= 5 for event in events if event.event_type == "movie_rate"))

    def test_laboratory_prequential_update_and_deduplication(self):
        simulation = SimulationEngine(SimulationConfig(
            user_count=2, interactions_per_user=2, random_seed=11,
            include_recommendation_events=False,
        ))
        events = simulation.run_batch(["Movie A", "Movie B", "Movie C"])
        laboratory = LaboratoryEngine()
        for event in events:
            laboratory.consume(event)
        before = laboratory.metrics()
        self.assertEqual(before["model_updates"], 4)
        self.assertEqual(before["rating_count"], 4)
        self.assertIsNotNone(before["rmse"])
        self.assertIsNotNone(before["mae"])
        laboratory.consume(events[-1])
        after = laboratory.metrics()
        self.assertEqual(after["processed_events"], before["processed_events"])
        self.assertEqual(after["model_updates"], before["model_updates"])

    def test_empty_catalog_is_rejected(self):
        simulation = SimulationEngine(SimulationConfig(user_count=1))
        with self.assertRaises(ValueError):
            simulation.run_batch([])


if __name__ == "__main__":
    unittest.main()

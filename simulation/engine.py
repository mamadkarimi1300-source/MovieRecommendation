"""Batch simulation engine that emits reproducible synthetic interaction events."""
from simulation.behavior.decision import BehaviorEngine
from simulation.config import SimulationConfig
from simulation.events.schema import InteractionEvent
from simulation.users.generator import UserGenerator


class SimulationEngine:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.users = UserGenerator(self.config.random_seed).generate(self.config.user_count)
        self.behavior = BehaviorEngine(self.config.random_seed + 1)

    def run_batch(self, catalog, recommendation_titles=None, on_event=None):
        catalog = [str(title) for title in catalog if str(title).strip()]
        if not catalog:
            raise ValueError("Movie catalog is empty; load MovieLens ratings first.")
        recommendations = [str(x) for x in (recommendation_titles or []) if str(x) in catalog]
        events = []
        for user in self.users:
            seen = set()
            for _ in range(self.config.interactions_per_user):
                title = self.behavior.choose_movie(user, catalog, seen)
                seen.add(title)
                if self.config.include_recommendation_events and recommendations:
                    rec_title = recommendations[int(self.behavior.rng.integers(0, len(recommendations)))]
                    events.append(InteractionEvent.create("recommendation_view", user.user_id, rec_title))
                    if self.behavior.rng.random() < 0.55:
                        events.append(InteractionEvent.create("recommendation_click", user.user_id, rec_title))
                events.append(InteractionEvent.create("movie_view", user.user_id, title))
                rating = self.behavior.make_rating(user, title)
                events.append(InteractionEvent.create("movie_rate", user.user_id, title, rating))
        if on_event:
            for event in events:
                on_event(event)
        return events

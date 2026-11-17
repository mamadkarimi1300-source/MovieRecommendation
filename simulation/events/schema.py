"""Validated event representation shared by Simulation and Laboratory."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True)
class InteractionEvent:
    event_id: str
    event_type: str
    user_id: str
    movie_title: str
    timestamp: str
    source: str = "simulation"
    rating: float | None = None

    @classmethod
    def create(cls, event_type: str, user_id: str, movie_title: str, rating=None):
        allowed = {"movie_view", "movie_rate", "recommendation_view", "recommendation_click"}
        if event_type not in allowed:
            raise ValueError(f"Unsupported event type: {event_type}")
        if event_type == "movie_rate" and (rating is None or not 1 <= float(rating) <= 5):
            raise ValueError("movie_rate events require a rating between 1 and 5")
        return cls(str(uuid4()), event_type, str(user_id), str(movie_title),
                   datetime.now(timezone.utc).isoformat(), "simulation",
                   None if rating is None else float(rating))

    def to_dict(self):
        return asdict(self)

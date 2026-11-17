"""Consumes simulation events and tracks prequential learning metrics."""
import math
from collections import Counter
from laboratory.learning.incremental import IncrementalMatrixFactorization


class LaboratoryEngine:
    def __init__(self, model=None):
        self.model = model or IncrementalMatrixFactorization()
        self.seen_event_ids = set()
        self.event_counts = Counter()
        self.absolute_errors = []
        self.squared_errors = []
        self.processed_events = 0
        self.model_updates = 0

    def consume(self, event):
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        event_id = data.get("event_id")
        if not event_id or event_id in self.seen_event_ids:
            return None
        self.seen_event_ids.add(event_id)
        self.event_counts[data["event_type"]] += 1
        self.processed_events += 1
        if data["event_type"] != "movie_rate":
            return None
        rating = float(data["rating"])
        prediction = self.model.predict(data["user_id"], data["movie_title"])
        error = rating - prediction
        self.absolute_errors.append(abs(error))
        self.squared_errors.append(error * error)
        self.model.update(data["user_id"], data["movie_title"], rating)
        self.model_updates += 1
        return {"prediction": prediction, "actual": rating, "error": error}

    def metrics(self):
        count = len(self.absolute_errors)
        return {
            "processed_events": self.processed_events,
            "model_updates": self.model_updates,
            "rating_count": count,
            "rmse": math.sqrt(sum(self.squared_errors) / count) if count else None,
            "mae": sum(self.absolute_errors) / count if count else None,
            "event_counts": dict(self.event_counts),
        }

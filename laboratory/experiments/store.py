"""Small JSON experiment-history store."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ExperimentStore:
    def __init__(self, path="artifacts/experiments/history.json"):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, config, metrics, status="completed"):
        records = self._read()
        record = {
            "experiment_id": "exp_" + uuid4().hex[:10],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "config": config,
            "metrics": metrics,
        }
        records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def list(self):
        return self._read()

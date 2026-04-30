from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cvcap.core.config import RunnerArgs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"
MODELS_DIR = PROJECT_ROOT / "models"


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    def load(self) -> RunnerArgs:
        defaults = RunnerArgs()
        if not self.path.exists():
            self.save(defaults)
            return defaults

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.save(defaults)
            return defaults

        merged = asdict(defaults)
        merged.update({key: value for key, value in payload.items() if key in merged})
        merged["yolo_classes"] = self._normalize_yolo_classes(merged.get("yolo_classes"))
        return RunnerArgs(**merged)

    def save(self, config: RunnerArgs) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(config)
        data["yolo_classes"] = list(config.yolo_classes) if config.yolo_classes is not None else None
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def available_models(self) -> list[str]:
        if not MODELS_DIR.exists():
            return []
        models: list[str] = []
        for item in sorted(MODELS_DIR.iterdir()):
            if item.is_file() and item.suffix.lower() in {".pt", ".onnx", ".engine"} and item.name.lower().startswith("yolo26"):
                models.append(str(Path("models") / item.name).replace("\\", "/"))
        return models

    @staticmethod
    def _normalize_yolo_classes(value: Any):
        if value in (None, "", "none", "all"):
            return None if value in ("", "none", "all") else value
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return tuple(int(part) for part in parts) if parts else None
        if isinstance(value, (list, tuple)):
            return tuple(int(part) for part in value)
        return value

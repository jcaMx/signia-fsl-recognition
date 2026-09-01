from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from core.model_catalog import ModelCatalog


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class PredictionContext:
    frame: Any
    mediapipe_result: Any
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    prediction: str
    confidence: float
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        response = {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "mode": self.mode,
        }
        response.update(self.metadata)
        return response


class BasePredictor(ABC):
    mode: str

    @abstractmethod
    def predict(self, context: PredictionContext) -> PredictionResponse:
        raise NotImplementedError


class ModelManager:
    def __init__(self, models_package: str = "models") -> None:
        self.models_package = models_package
        self.models_path = Path(__file__).resolve().parent.parent / "models"
        self._predictors: dict[str, BasePredictor] = {}
        self._discovered = False

    def register(self, predictor: BasePredictor) -> None:
        mode = predictor.mode.strip().lower()
        self._predictors[mode] = predictor

    def get(self, mode: str) -> Optional[BasePredictor]:
        return self._predictors.get(mode.strip().lower())

    def available_modes(self) -> list[str]:
        return sorted(self._predictors.keys())

    def discover_models(self) -> None:
        if self._discovered:
            return

        for predictor_path in self.models_path.glob("*/predictor.py"):
            package_name = predictor_path.parent.name
            module = import_module(f"{self.models_package}.{package_name}.predictor")
            register = getattr(module, "register", None)
            if callable(register):
                register(self)

        # Also discover dynamically trained models from artifacts/models
        try:
            catalog = ModelCatalog()
            cat_data = catalog.get_catalog()
            models_dict = cat_data.get("models", {})
            categories_dict = cat_data.get("categories", {})
            labels_map = catalog.labels_map

            for cat, rel_path in models_dict.items():
                pt_path = catalog.project_dir / rel_path
                if pt_path.exists():
                    try:
                        from core.artifact_predictor import ArtifactPredictor
                        predictor = ArtifactPredictor(category=cat, model_path=pt_path, labels_map=labels_map)
                        self.register(predictor)
                        print(f"Registered ArtifactPredictor for {cat} at {pt_path.name}")
                    except Exception as e:
                        print(f"Failed to register ArtifactPredictor for {cat}: {e}")
        except Exception as e:
            print(f"Failed to scan artifact models during discover_models: {e}")

        self._discovered = True

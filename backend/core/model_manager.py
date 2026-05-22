from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Optional


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

        self._discovered = True

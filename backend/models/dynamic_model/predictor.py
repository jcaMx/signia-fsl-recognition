from __future__ import annotations

from core.model_manager import BasePredictor, ModelManager, PredictionContext, PredictionError


class DynamicPredictor(BasePredictor):
    mode = "dynamic"

    def predict(self, context: PredictionContext):
        raise PredictionError("Mode 'dynamic' is registered but not implemented yet.", status_code=501)


def register(model_manager: ModelManager) -> None:
    model_manager.register(DynamicPredictor())

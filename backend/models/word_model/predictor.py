from __future__ import annotations

from core.model_manager import BasePredictor, ModelManager, PredictionContext, PredictionError


class WordPredictor(BasePredictor):
    mode = "word"

    def predict(self, context: PredictionContext):
        raise PredictionError("Mode 'word' is registered but not implemented yet.", status_code=501)


def register(model_manager: ModelManager) -> None:
    model_manager.register(WordPredictor())

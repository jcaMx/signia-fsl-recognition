from __future__ import annotations

from core.model_manager import BasePredictor, ModelManager, PredictionContext, PredictionError


class PhrasePredictor(BasePredictor):
    mode = "phrase"

    def predict(self, context: PredictionContext):
        raise PredictionError("Mode 'phrase' is registered but not implemented yet.", status_code=501)


def register(model_manager: ModelManager) -> None:
    model_manager.register(PhrasePredictor())

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core.model_manager import (
    BasePredictor,
    ModelManager,
    PredictionContext,
    PredictionError,
    PredictionResponse,
)
from core.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    build_single_frame_input,
    flatten_xyz_landmarks,
    get_hand_count,
    get_first_hand_landmarks,
    normalize_landmarks,
)


class AlphabetPredictor(BasePredictor):
    mode = "alphabet"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
        preprocess_config_path: Optional[Path] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.model_path = model_path or (project_root / "models" / "static_fsl_model.keras")
        self.label_encoder_path = label_encoder_path or (project_root / "models" / "label_encoder.pkl")
        self.preprocess_config_path = preprocess_config_path or (
            project_root / "models" / "preprocess_config.pkl"
        )
        self._model = None
        self._label_encoder = None
        self._preprocess_config = None

    def _load_model(self):
        if self._model is None:
            from keras.models import load_model

            self._model = load_model(self.model_path)
        return self._model

    def _load_label_encoder(self):
        if self._label_encoder is None:
            with self.label_encoder_path.open("rb") as file:
                self._label_encoder = pickle.load(file)
        return self._label_encoder

    def _load_preprocess_config(self) -> dict:
        if self._preprocess_config is None:
            with self.preprocess_config_path.open("rb") as file:
                self._preprocess_config = pickle.load(file)
        return self._preprocess_config

    def predict(self, context: PredictionContext) -> PredictionResponse:
        hand_landmarks = get_first_hand_landmarks(context.mediapipe_result)
        if hand_landmarks is None:
            return PredictionResponse(
                prediction="",
                confidence=0.0,
                mode=self.mode,
                metadata={
                    "hand_detected": False,
                    "hand_count": get_hand_count(context.mediapipe_result),
                },
            )

        try:
            raw_features = flatten_xyz_landmarks(hand_landmarks)
            preprocess_config = self._load_preprocess_config()
            scale_mode = preprocess_config.get("scale_mode", "bbox")
            features = normalize_landmarks(raw_features, scale_mode=scale_mode)
            input_data = build_single_frame_input(
                features,
                expected_feature_count=self._expected_feature_count(),
            )
        except ValueError as error:
            raise PredictionError(str(error), status_code=500) from error

        prediction_scores = self._load_model().predict(input_data, verbose=0)[0]
        predicted_index = int(np.argmax(prediction_scores))
        confidence = float(np.max(prediction_scores))
        predicted_label = self._load_label_encoder().inverse_transform([predicted_index])[0]

        return PredictionResponse(
            prediction=str(predicted_label),
            confidence=confidence,
            mode=self.mode,
            metadata={
                "hand_detected": True,
                "hand_count": get_hand_count(context.mediapipe_result),
                "landmark_count": len(hand_landmarks.landmark),
                "feature_count": len(features),
                "model_input_features": self._expected_feature_count(),
                "preprocess_scale_mode": scale_mode,
            },
        )

    def _expected_feature_count(self) -> int:
        model = self._load_model()
        input_shape = getattr(model, "input_shape", None)
        if not input_shape or input_shape[-1] is None:
            return EXPECTED_FEATURE_COUNT
        return int(input_shape[-1])


def register(model_manager: ModelManager) -> None:
    model_manager.register(AlphabetPredictor())

from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from core.model_manager import ModelManager, PredictionContext, PredictionError
from core.prediction_stabilizer import PredictionStabilizer
from webcam.camera_manager import CameraManager
from webcam.mediapipe_pipeline import MediaPipeHandsPipeline


class PredictionRouter:
    def __init__(
        self,
        model_manager: ModelManager,
        camera_manager: CameraManager,
        mediapipe_pipeline: MediaPipeHandsPipeline,
        prediction_stabilizer: PredictionStabilizer,
    ) -> None:
        self.model_manager = model_manager
        self.camera_manager = camera_manager
        self.mediapipe_pipeline = mediapipe_pipeline
        self.prediction_stabilizer = prediction_stabilizer
        self.blueprint = Blueprint("prediction", __name__)
        self._register_routes()

    def _register_routes(self) -> None:
        @self.blueprint.get("/")
        def home():
            return jsonify(
                {
                    "status": "running",
                    "available_modes": self.model_manager.available_modes(),
                }
            )

        @self.blueprint.get("/modes")
        def modes():
            return jsonify({"modes": self.model_manager.available_modes()})

        @self.blueprint.post("/predict")
        def predict():
            payload = request.get_json(silent=True) or {}
            mode = payload.get("mode", "alphabet")
            return self._predict_for_mode(mode, payload)

        @self.blueprint.get("/predict_letter")
        def predict_letter():
            return self._predict_for_mode("alphabet", {"mode": "alphabet"})

    def _predict_for_mode(self, mode: str, payload: dict):
        predictor = self.model_manager.get(mode)
        if predictor is None:
            return (
                jsonify(
                    {
                        "error": f"Unsupported mode '{mode}'.",
                        "available_modes": self.model_manager.available_modes(),
                    }
                ),
                404,
            )

        frame_source = self._resolve_frame_source(payload)
        frame = self._load_frame(frame_source=frame_source, payload=payload)
        if frame is None:
            if frame_source == "client":
                return (
                    jsonify(
                        {
                            "error": "Client image not provided",
                            "details": "Send a valid base64 data URL in the 'image' field when frame_source is 'client'.",
                        }
                    ),
                    400,
                )

            return (
                jsonify(
                    {
                        "error": "Camera not accessible",
                        "details": "No client image was provided, so the backend attempted to read from the server webcam.",
                    }
                ),
                503,
            )

        mediapipe_result = self.mediapipe_pipeline.process(frame)
        context = PredictionContext(
            frame=frame,
            mediapipe_result=mediapipe_result,
            payload=payload,
        )

        try:
            response = predictor.predict(context)
        except PredictionError as error:
            return jsonify({"error": error.message, "mode": mode}), error.status_code

        stream_id = self._build_stream_id(mode=mode, payload=payload)
        should_stabilize = self._should_stabilize(payload)
        if should_stabilize:
            response = self.prediction_stabilizer.stabilize(stream_id=stream_id, response=response)
        else:
            response.metadata.setdefault("raw_prediction", response.prediction)
            response.metadata.setdefault("raw_confidence", response.confidence)
            response.metadata.setdefault("stabilized_prediction", response.prediction)
            response.metadata.setdefault("stabilized_confidence", response.confidence)
            response.metadata.setdefault("stabilized_count", 1 if response.prediction else 0)
            response.metadata.setdefault("history_length", 0)
            response.metadata.setdefault("stabilized", bool(response.prediction))
            response.metadata.setdefault("emitted", bool(response.prediction))
            response.metadata.setdefault("suppressed_change", False)
            response.metadata.setdefault("cooldown_remaining_ms", 0)
            response.metadata.setdefault("confidence_threshold", 0.0)

        response.metadata.setdefault("stream_id", stream_id)
        response.metadata.setdefault("frame_source", frame_source)
        response.metadata.setdefault("stabilization_enabled", should_stabilize)
        response.metadata.setdefault("frame_shape", list(frame.shape) if frame is not None else None)
        return jsonify(response.to_dict())

    def _load_frame(self, frame_source: str, payload: dict):
        if frame_source == "client":
            return self._extract_frame_from_payload(payload)

        success, frame = self.camera_manager.read_frame()
        if not success:
            return None
        return frame

    def _extract_frame_from_payload(self, payload: dict):
        image_data_url = payload.get("image")
        if not image_data_url or not isinstance(image_data_url, str):
            return None

        if "," not in image_data_url:
            return None

        try:
            encoded_bytes = image_data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded_bytes)
            np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        except (ValueError, TypeError):
            return None

        return frame

    def _build_stream_id(self, mode: str, payload: dict) -> str:
        client_key = (
            payload.get("stream_id")
            or payload.get("session_id")
            or payload.get("client_id")
            or request.remote_addr
            or "anonymous"
        )
        return f"{mode}:{client_key}"

    def _resolve_frame_source(self, payload: dict) -> str:
        requested_source = str(payload.get("frame_source", "")).strip().lower()
        if requested_source in {"client", "server"}:
            return requested_source

        image_data = payload.get("image")
        if isinstance(image_data, str) and image_data:
            return "client"

        return "server"

    def _should_stabilize(self, payload: dict) -> bool:
        explicit_value = payload.get("stabilize")
        if explicit_value is not None:
            return bool(explicit_value)

        frame_source = self._resolve_frame_source(payload)
        if frame_source == "client":
            return False

        return True

from __future__ import annotations

import cv2
import mediapipe as mp
from flask import Flask, Response, render_template, request, stream_with_context
from flask_cors import CORS

from core.model_manager import ModelManager, PredictionContext, PredictionError
from core.prediction_router import PredictionRouter
from core.prediction_stabilizer import PredictionStabilizer, PredictionStabilizerConfig
from core.preprocessing import get_hand_count
from webcam.camera_manager import CameraManager
from webcam.mediapipe_pipeline import MediaPipeHandsPipeline


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    model_manager = ModelManager()
    model_manager.discover_models()

    camera_manager = CameraManager()
    mediapipe_pipeline = MediaPipeHandsPipeline()
    prediction_stabilizer = PredictionStabilizer(PredictionStabilizerConfig.from_env())
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    prediction_router = PredictionRouter(
        model_manager=model_manager,
        camera_manager=camera_manager,
        mediapipe_pipeline=mediapipe_pipeline,
        prediction_stabilizer=prediction_stabilizer,
    )

    @app.get("/test")
    def test_page():
        return render_template("test.html")

    @app.get("/debug_feed")
    def debug_feed():
        mode = request.args.get("mode", "alphabet")
        predictor = model_manager.get(mode)

        def generate():
            while True:
                success, frame = camera_manager.read_frame()
                if not success or frame is None:
                    continue

                mediapipe_result = mediapipe_pipeline.process(frame)
                hand_detected = bool(getattr(mediapipe_result, "multi_hand_landmarks", None))
                hand_count = get_hand_count(mediapipe_result)

                if hand_detected:
                    for hand_landmarks in mediapipe_result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                raw_prediction = ""
                confidence = 0.0
                if predictor is not None:
                    context = PredictionContext(
                        frame=frame,
                        mediapipe_result=mediapipe_result,
                        payload={"mode": mode, "frame_source": "server"},
                    )
                    try:
                        prediction_response = predictor.predict(context)
                        raw_prediction = prediction_response.prediction
                        confidence = prediction_response.confidence
                    except PredictionError:
                        raw_prediction = ""
                        confidence = 0.0

                overlays = [
                    f"mode: {mode}",
                    f"hand_detected: {hand_detected}",
                    f"hand_count: {hand_count}",
                    f"raw_prediction: {raw_prediction or 'None'}",
                    f"confidence: {confidence:.3f}",
                ]

                for index, text in enumerate(overlays):
                    y = 30 + (index * 28)
                    cv2.putText(
                        frame,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                success, encoded_frame = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded_frame.tobytes() + b"\r\n"
                )

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    app.register_blueprint(prediction_router.blueprint)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

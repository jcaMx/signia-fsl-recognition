from flask import Flask, render_template
from flask_cors import CORS

from core.model_manager import ModelManager
from core.prediction_router import PredictionRouter
from core.prediction_stabilizer import PredictionStabilizer, PredictionStabilizerConfig
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

    prediction_router = PredictionRouter(
        model_manager=model_manager,
        camera_manager=camera_manager,
        mediapipe_pipeline=mediapipe_pipeline,
        prediction_stabilizer=prediction_stabilizer,
    )

    @app.get("/test")
    def test_page():
        return render_template("test.html")

    app.register_blueprint(prediction_router.blueprint)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

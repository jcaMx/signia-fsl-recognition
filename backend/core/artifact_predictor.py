from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Deque, Any

import numpy as np

from core.model_manager import (
    BasePredictor,
    PredictionContext,
    PredictionError,
    PredictionResponse,
)
from core.preprocessing import (
    flatten_xyz_landmarks,
    get_first_hand_landmarks,
    get_hand_count,
)

logger = logging.getLogger(__name__)

# To load the PyTorch model and preprocessing functions
import sys
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.sign_lstm import SignLSTM
from src.preprocessing.greeting_features import add_motion_features, normalize_sequence

class ArtifactPredictor(BasePredictor):
    def __init__(self, category: str, model_path: Path, labels_map: Dict[int, str]):
        self.mode = category.lower()
        self.category = category
        self.model_path = model_path
        self.labels_map = labels_map

        self._model: Optional[Any] = None
        self._input_size: int = 126
        self._num_classes: int = 0
        
        self._lock = Lock()
        # buffer for frames per stream_id. value is a deque of numpy arrays
        self._buffers: Dict[str, Deque[np.ndarray]] = {}
        self._SEQUENCE_LENGTH = 30

        self._load_model()

    def _load_model(self):
        import torch
        from src.models.sign_lstm import SignLSTM
        
        logger.info(f"Loading artifact checkpoint: {self.model_path}")
        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
            
            self._input_size = int(checkpoint["input_size"])
            self._num_classes = int(checkpoint["num_classes"])
            
            self._model = SignLSTM(
                input_size=self._input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=self._num_classes,
                dropout=0.3,
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()

            original_ids = list(checkpoint["original_label_ids"])
            self._local_labels = {}
            for argmax_idx, label_id in enumerate(original_ids):
                name = self.labels_map.get(int(label_id), f"Class {label_id}")
                self._local_labels[argmax_idx] = name

        except Exception as e:
            logger.error(f"Failed to load artifact model {self.category}: {e}")
            raise RuntimeError(f"Could not load model {self.model_path}") from e

    def _get_stream_id(self, context: PredictionContext) -> str:
        payload = context.payload
        # Follow the same logic as PredictionRouter._build_stream_id (which sets stream_id in metadata)
        # However, prediction_router passes payload containing stream_id/session_id
        return str(
            payload.get("stream_id")
            or payload.get("session_id")
            or payload.get("client_id")
            or "anonymous"
        )

    def _preprocess_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        # frames is a list of shape (126,)
        seq = np.array(frames) # (30, 126)
        
        # normalize
        seq = normalize_sequence(seq)
        
        if self._input_size == 252:
            seq = add_motion_features(seq)
            
        return seq

    def predict(self, context: PredictionContext) -> PredictionResponse:
        stream_id = self._get_stream_id(context)

        hand_landmarks = get_first_hand_landmarks(context.mediapipe_result)
        if hand_landmarks is None:
            # We don't add zeroes if no hand is detected
            return PredictionResponse(
                prediction="",
                confidence=0.0,
                mode=self.mode,
                metadata={
                    "hand_detected": False,
                    "hand_count": get_hand_count(context.mediapipe_result),
                    "buffer_size": len(self._buffers.get(stream_id, [])),
                },
            )
            
        try:
            # Flatten landmarks to 126 (42 points * 3 dims)
            # Actually, standard flatten is (21 landmarks * 3 dims * 2 hands) -> wait, get_first_hand_landmarks returns ONE hand.
            # But the backend's flatten_xyz_landmarks might handle it.
            # wait, flatten_xyz_landmarks in preprocessing flattens all landmarks.
            raw_features = flatten_xyz_landmarks(hand_landmarks) 
            # In our data collection for artifacts, each frame is usually 126 elements. (2 hands * 21 * 3 = 126). 
            # Wait, get_first_hand_landmarks returns ONE hand, so flattening it gives 63 features. 
            # If the model expects 126 features per frame, we might need both hands.
            # Let's extract exactly as the collection script did. 
            # The mediapipe_result contains all hands.
            
            # Let's rebuild the 126-feature array:
            # MediaPipe hands can have up to 2 hands.
            points = np.zeros(126, dtype=np.float32)
            if context.mediapipe_result and getattr(context.mediapipe_result, "multi_hand_landmarks", None):
                landmarks_list = context.mediapipe_result.multi_hand_landmarks
                
                # Hand 1
                if len(landmarks_list) > 0:
                    idx = 0
                    for lm in landmarks_list[0].landmark:
                        points[idx] = lm.x
                        points[idx+1] = lm.y
                        points[idx+2] = lm.z
                        idx += 3
                # Hand 2
                if len(landmarks_list) > 1:
                    idx = 63
                    for lm in landmarks_list[1].landmark:
                        points[idx] = lm.x
                        points[idx+1] = lm.y
                        points[idx+2] = lm.z
                        idx += 3
            
        except Exception as error:
            raise PredictionError(str(error), status_code=500) from error

        with self._lock:
            if stream_id not in self._buffers:
                self._buffers[stream_id] = deque(maxlen=self._SEQUENCE_LENGTH)
            
            self._buffers[stream_id].append(points)
            buffer_len = len(self._buffers[stream_id])
            
            if buffer_len < self._SEQUENCE_LENGTH:
                # Need more frames
                return PredictionResponse(
                    prediction="",
                    confidence=0.0,
                    mode=self.mode,
                    metadata={
                        "hand_detected": True,
                        "hand_count": get_hand_count(context.mediapipe_result),
                        "buffer_size": buffer_len,
                        "message": f"Buffering... ({buffer_len}/{self._SEQUENCE_LENGTH})",
                    },
                )
                
            # We have 30 frames
            seq_list = list(self._buffers[stream_id])
            
        # Preprocess and run inference
        try:
            import torch
            processed_seq = self._preprocess_sequence(seq_list)
            # Add batch dim -> (1, 30, input_size)
            tensor_input = torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                out = self._model(tensor_input)
                probs = torch.softmax(out, dim=1).squeeze(0)
                confidence, pred_idx = torch.max(probs, dim=0)
                
                pred_val = confidence.item()
                idx_val = pred_idx.item()
                
            predicted_label = self._local_labels.get(idx_val, "")
        except Exception as e:
            logger.error(f"Inference error in {self.category}: {e}")
            raise PredictionError("Inference failed", status_code=500) from e

        return PredictionResponse(
            prediction=str(predicted_label),
            confidence=float(pred_val),
            mode=self.mode,
            metadata={
                "hand_detected": True,
                "hand_count": get_hand_count(context.mediapipe_result),
                "buffer_size": buffer_len,
                "model_input_size": self._input_size,
            },
        )

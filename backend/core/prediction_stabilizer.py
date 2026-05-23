from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Optional

from core.model_manager import PredictionResponse


@dataclass(frozen=True)
class PredictionStabilizerConfig:
    history_size: int = 10
    stability_window_ms: int = 3000
    confidence_threshold: float = 0.60
    min_consistent_frames: int = 2
    cooldown_ms: int = 300
    change_suppression_ms: int = 0
    stale_ttl_ms: int = 5000

    @classmethod
    def from_env(cls) -> "PredictionStabilizerConfig":
        return cls(
            history_size=int(os.getenv("FSL_STABILIZER_HISTORY_SIZE", cls.history_size)),
            stability_window_ms=int(
                os.getenv("FSL_STABILIZER_WINDOW_MS", cls.stability_window_ms)
            ),
            confidence_threshold=float(
                os.getenv(
                    "FSL_STABILIZER_CONFIDENCE_THRESHOLD",
                    cls.confidence_threshold,
                )
            ),
            min_consistent_frames=int(
                os.getenv(
                    "FSL_STABILIZER_MIN_CONSISTENT_FRAMES",
                    cls.min_consistent_frames,
                )
            ),
            cooldown_ms=int(os.getenv("FSL_STABILIZER_COOLDOWN_MS", cls.cooldown_ms)),
            change_suppression_ms=int(
                os.getenv(
                    "FSL_STABILIZER_CHANGE_SUPPRESSION_MS",
                    cls.change_suppression_ms,
                )
            ),
            stale_ttl_ms=int(os.getenv("FSL_STABILIZER_STALE_TTL_MS", cls.stale_ttl_ms)),
        )


@dataclass
class PredictionObservation:
    label: str
    confidence: float
    timestamp_ms: int


@dataclass
class StreamState:
    history: Deque[PredictionObservation] = field(default_factory=deque)
    last_observation_at_ms: int = 0
    last_emitted_prediction: str = ""
    last_emitted_confidence: float = 0.0
    last_emitted_at_ms: int = 0


class PredictionStabilizer:
    def __init__(self, config: Optional[PredictionStabilizerConfig] = None) -> None:
        self.config = config or PredictionStabilizerConfig()
        self._states: dict[str, StreamState] = {}
        self._lock = Lock()

    def stabilize(
        self,
        stream_id: str,
        response: PredictionResponse,
        now_ms: Optional[int] = None,
    ) -> PredictionResponse:
        timestamp_ms = now_ms if now_ms is not None else self._now_ms()
        raw_prediction = response.prediction
        raw_confidence = response.confidence

        with self._lock:
            state = self._states.setdefault(stream_id, StreamState())

            print(
                "STABILIZER DEBUG",
                {
                    "stream_id": stream_id,
                    "state_count": len(self._states),
                    "history_before": len(state.history),
                    "raw_prediction": raw_prediction,
                    "raw_confidence": raw_confidence,
                },
            )

            print(
                "STABILIZER AFTER",
                {
                    "stream_id": stream_id,
                    "history_after": len(state.history),
                    "history_labels": [item.label for item in state.history],
                },
            )

            if raw_prediction and raw_confidence >= self.config.confidence_threshold:
                state.history.append(
                    PredictionObservation(
                        label=raw_prediction,
                        confidence=raw_confidence,
                        timestamp_ms=timestamp_ms,
                    )
                )
                state.last_observation_at_ms = timestamp_ms

            # self._prune_history(state, timestamp_ms)

            if self._is_stale(state, timestamp_ms):
                state.last_emitted_prediction = ""
                state.last_emitted_confidence = 0.0
                state.last_emitted_at_ms = 0
                state.history.clear()

            candidate_label, candidate_confidence, candidate_count = self._stable_candidate(
                state.history
            )

            suppressed_change = False
            if (
                candidate_label
                and state.last_emitted_prediction
                and candidate_label != state.last_emitted_prediction
                and timestamp_ms - state.last_emitted_at_ms < self.config.change_suppression_ms
            ):
                suppressed_change = True
                candidate_label = state.last_emitted_prediction
                candidate_confidence = state.last_emitted_confidence

            emitted = False
            if candidate_label and timestamp_ms - state.last_emitted_at_ms >= self.config.cooldown_ms:
                state.last_emitted_prediction = candidate_label
                state.last_emitted_confidence = candidate_confidence
                state.last_emitted_at_ms = timestamp_ms
                emitted = True

            output_prediction = state.last_emitted_prediction if state.last_emitted_prediction else ""
            output_confidence = (
                state.last_emitted_confidence if state.last_emitted_prediction else 0.0
            )
            cooldown_remaining_ms = max(
                0,
                self.config.cooldown_ms - (timestamp_ms - state.last_emitted_at_ms),
            )

        metadata = dict(response.metadata)
        metadata.update(
            {
                "raw_prediction": raw_prediction,
                "raw_confidence": raw_confidence,
                "stabilized_prediction": candidate_label or "",
                "stabilized_confidence": candidate_confidence,
                "stabilized_count": candidate_count,
                "stabilized": bool(candidate_label),
                "emitted": emitted,
                "suppressed_change": suppressed_change,
                "cooldown_remaining_ms": cooldown_remaining_ms,
                "confidence_threshold": self.config.confidence_threshold,
                "history_length": len(state.history),
            }
        )

        return PredictionResponse(
            prediction=output_prediction,
            confidence=output_confidence,
            mode=response.mode,
            metadata=metadata,
        )

    def _stable_candidate(
        self,
        history: Deque[PredictionObservation],
    ) -> tuple[str, float, int]:
        if len(history) < self.config.min_consistent_frames:
            return "", 0.0, 0

        counts = Counter(observation.label for observation in history)
        label, count = counts.most_common(1)[0]
        if count < self.config.min_consistent_frames:
            return "", 0.0, 0

        confidences = [
            observation.confidence for observation in history if observation.label == label
        ]
        confidence = sum(confidences) / len(confidences)
        return label, confidence, count

    def _prune_history(self, state: StreamState, now_ms: int) -> None:
        min_timestamp = now_ms - self.config.stability_window_ms
        while state.history and (
            len(state.history) > self.config.history_size
            or state.history[0].timestamp_ms < min_timestamp
        ):
            state.history.popleft()

    def _is_stale(self, state: StreamState, now_ms: int) -> bool:
        if state.last_observation_at_ms == 0:
            return False
        return now_ms - state.last_observation_at_ms > self.config.stale_ttl_ms

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

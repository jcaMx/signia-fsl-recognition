import sys
import os
import cv2
import time
import logging

import mediapipe as mp


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

sys.path.append(PROJECT_ROOT)


from src.data_collection.collector import SequenceCollector
from src.data_collection.collection_stats import CollectionStats


# ============================================================
# MediaPipe Hands
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# Main
# ============================================================

def main():

    logger.info("========================================")
    logger.info(" FSL Sequence Data Collector (CLI Demo) ")
    logger.info("========================================")

    category = input(
        "Enter category (e.g., GREETING): "
    ).strip().upper()

    label = input(
        "Enter label (e.g., GOOD MORNING): "
    ).strip().upper()

    if not category or not label:
        logger.error(
            "Category and label cannot be empty."
        )
        return

    stats = CollectionStats()

    logger.info("\nCurrent collection status:")
    print(stats.get_stats_summary())
    print()

    logger.info(
        f"Initializing webcam for {label} ({category})..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info(
        "Press 'r' to start recording a sequence."
    )

    logger.info(
        "Press 'q' to quit."
    )

    # ========================================================
    # UI state
    # ========================================================

    transient_message = ""
    transient_message_time = 0

    MESSAGE_DURATION = 2.0

    # ========================================================
    # MediaPipe Hands
    # ========================================================

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ========================================================
    # Collector
    # ========================================================

    try:

        with SequenceCollector() as collector:

            recording = False

            while True:

                ret, frame = cap.read()

                if not ret:

                    logger.error(
                        "Failed to grab frame."
                    )

                    break

                # ------------------------------------------------
                # Flip webcam for mirror-like display
                # ------------------------------------------------

                frame_display = cv2.flip(
                    frame,
                    1
                )

                # ------------------------------------------------
                # MediaPipe processing
                # ------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame_display,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(
                    rgb_frame
                )

                # ------------------------------------------------
                # Draw hand landmarks
                # ------------------------------------------------

                detected_hands = []

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(
                            frame_display,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                # ------------------------------------------------
                # Determine detected left/right hands
                # ------------------------------------------------

                left_detected = False
                right_detected = False

                if results.multi_handedness:

                    for handedness in results.multi_handedness:

                        hand_label = handedness.classification[0].label

                        if hand_label == "Left":
                            left_detected = True

                        elif hand_label == "Right":
                            right_detected = True

                # ------------------------------------------------
                # Tracking status
                # ------------------------------------------------

                tracking_text = (
                    f"Hands: "
                    f"L={'OK' if left_detected else '--'} "
                    f"R={'OK' if right_detected else '--'}"
                )

                cv2.putText(
                    frame_display,
                    tracking_text,
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Transient messages
                # ------------------------------------------------

                if (
                    time.time() - transient_message_time
                    < MESSAGE_DURATION
                ):

                    cv2.putText(
                        frame_display,
                        transient_message,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                # =================================================
                # Recording
                # =================================================

                if recording:

                    # IMPORTANT:
                    # Keep using the original frame here.
                    # SequenceCollector remains responsible for
                    # producing the saved (30, 126) sequence.

                    frames_collected = collector.process_frame(
                        frame
                    )

                    status_text = (
                        f"Recording: "
                        f"{frames_collected} / 30 frames"
                    )

                    color = (0, 0, 255)

                    # ------------------------------------------------
                    # Log progress
                    # ------------------------------------------------

                    if frames_collected > 0:

                        logger.info(
                            f"Captured frame "
                            f"{frames_collected}/30 "
                            f"for {label}"
                        )

                    # ------------------------------------------------
                    # Finalize sequence
                    # ------------------------------------------------

                    if frames_collected >= 30:

                        try:

                            filepath = (
                                collector.finalize_sequence(
                                    category,
                                    label
                                )
                            )

                            logger.info(
                                f"Sequence saved successfully "
                                f"to {filepath}"
                            )

                            transient_message = (
                                f"Saved: "
                                f"{os.path.basename(filepath)}"
                            )

                            transient_message_time = (
                                time.time()
                            )

                            recording = False

                            logger.info(
                                "\nUpdated Stats:"
                            )

                            print(
                                stats.get_stats_summary()
                            )

                            logger.info(
                                "Press 'r' to record another, "
                                "'q' to quit."
                            )

                        except Exception as e:

                            logger.error(
                                f"Error saving sequence: {e}"
                            )

                            transient_message = (
                                "Error saving sequence!"
                            )

                            transient_message_time = (
                                time.time()
                            )

                            recording = False

                else:

                    status_text = (
                        f"Ready: {label} "
                        f"(Press 'r' to Record, "
                        f"'q' to Quit)"
                    )

                    color = (0, 255, 0)

                # ------------------------------------------------
                # Status text
                # ------------------------------------------------

                cv2.putText(
                    frame_display,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                # ------------------------------------------------
                # Sample count
                # ------------------------------------------------

                current_count = (
                    stats.get_class_counts().get(
                        f"{category}/{label}",
                        0
                    )
                )

                count_text = (
                    f"Samples collected: "
                    f"{current_count}"
                )

                cv2.putText(
                    frame_display,
                    count_text,
                    (
                        10,
                        frame_display.shape[0] - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # ------------------------------------------------
                # Display webcam
                # ------------------------------------------------

                cv2.imshow(
                    "FSL Data Collector - Hand Tracking",
                    frame_display
                )

                # ------------------------------------------------
                # Keyboard
                # ------------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):

                    logger.info(
                        "Quitting collection..."
                    )

                    break

                elif (
                    key == ord('r')
                    and not recording
                ):

                    logger.info(
                        f"Starting recording for {label}..."
                    )

                    transient_message = (
                        "Recording Started!"
                    )

                    transient_message_time = (
                        time.time()
                    )

                    collector.start_recording()

                    recording = True

    finally:

        hands.close()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


    # python scripts/demo_collectionv1.py
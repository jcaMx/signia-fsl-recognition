"""
demo_collection.py
------------------
Phase 1 FSL Sequence Data Collector — Label-Driven CLI

Loads available categories and labels from csv/labels.csv via LabelRegistry.
No categories or labels are hardcoded here.

Usage:
    python scripts/demo_collection.py

Keyboard controls (webcam window):
    r  — start recording a sequence
    q  — quit / return to menu
"""

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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_collection.collector import SequenceCollector
from src.data_collection.collection_stats import CollectionStats
from src.data_collection.label_registry import LabelRegistry

# ============================================================
# MediaPipe
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================
# UI helpers
# ============================================================

DIVIDER = "=" * 44


def _print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f" {title}")
    print(DIVIDER)


def _numbered_menu(prompt: str, items: list[str]) -> int | None:
    """
    Print a numbered menu and return the 0-based index of the chosen item.

    Returns None if the user signals exit (empty input / Ctrl-C / EOF).
    Loops until valid input is received.
    """
    print(f"\n{prompt}\n")
    for i, item in enumerate(items, start=1):
        print(f"  {i}. {item}")
    print()

    while True:
        try:
            raw = input("Enter number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not raw:
            print("  Please enter a number.")
            continue

        if not raw.isdigit():
            print("  Invalid choice. Please enter a number.")
            continue

        choice = int(raw)
        if 1 <= choice <= len(items):
            return choice - 1  # 0-based

        print(f"  Out of range. Enter a number between 1 and {len(items)}.")


# ============================================================
# Selection menus
# ============================================================

def select_category(registry: LabelRegistry) -> str | None:
    """
    Display the category menu and return the selected category (UPPERCASE).
    Returns None if the user wants to exit.
    """
    _print_header("FSL Sequence Data Collector")
    categories = registry.get_categories()

    if not categories:
        logger.error("labels.csv has no categories. Cannot proceed.")
        return None

    idx = _numbered_menu("Select category:", categories)
    if idx is None:
        return None
    return categories[idx]


def select_label(registry: LabelRegistry, category: str) -> str | None:
    """
    Display the label menu for *category* and return the selected label (UPPERCASE).
    Returns None if the user wants to exit or if the category has no labels.
    """
    labels = registry.get_labels(category)

    if not labels:
        logger.warning("Category '%s' has no labels in labels.csv.", category)
        return None

    _print_header(f"Category: {category}")
    idx = _numbered_menu("Select label:", labels)
    if idx is None:
        return None
    return labels[idx]


# ============================================================
# Post-save menu
# ============================================================

def post_save_menu(label: str) -> int | None:
    """
    Ask the user what to do after a sequence is saved.

    Returns
    -------
    1 — record same label again
    2 — choose another label (same category)
    3 — choose another category
    4 — exit
    None — EOF / Ctrl-C (treat as exit)
    """
    _print_header("Sequence Saved")

    options = [
        f"Record another {label}",
        "Choose another label",
        "Choose another category",
        "Exit",
    ]
    print("\nWhat would you like to do?\n")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    print()

    while True:
        try:
            raw = input("Enter choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not raw:
            print("  Please enter a number.")
            continue

        if not raw.isdigit():
            print("  Invalid choice. Please enter a number.")
            continue

        choice = int(raw)
        if 1 <= choice <= 4:
            return choice

        print("  Out of range. Enter 1–4.")


# ============================================================
# Webcam collection session
# ============================================================

def run_collection_session(
    cap: cv2.VideoCapture,
    collector: SequenceCollector,
    stats: CollectionStats,
    hands: mp.solutions.hands.Hands,
    category: str,
    label: str,
) -> bool:
    """
    Run the webcam collection loop for a single category/label pair.

    The user presses 'r' to record and 'q' to quit/return to menu.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Already-opened capture device.
    collector : SequenceCollector
        Shared collector instance (manages frame buffer + saving).
    stats : CollectionStats
        Used to display live sample counts.
    hands : mp.solutions.hands.Hands
        MediaPipe hands solution for landmark drawing.
    category : str
        UPPERCASE category name.
    label : str
        UPPERCASE label name.

    Returns
    -------
    bool
        True if at least one sequence was saved.
        False if the user quit without saving.
    """
    # Build the stats key that CollectionStats uses
    # (e.g. "GREETING/GOOD MORNING" — uppercase, spaces not underscores)
    stats_key = f"{category}/{label}"

    # Reset any leftover frames from a previous session
    collector.start_recording()

    recording = False
    saved_count = 0

    transient_message = ""
    transient_message_time = 0.0
    MESSAGE_DURATION = 2.0

    # --------------------------------------------------
    # Print selection summary to console
    # --------------------------------------------------
    _print_header("Selected")
    print(f"\n  Category : {category}")
    print(f"  Label    : {label}")
    current_count = stats.get_class_counts().get(stats_key, 0)
    print(f"\n  Sequences collected: {current_count}")
    print("\n  Press 'r' in webcam window to record.")
    print("  Press 'q' in webcam window to quit/return to menu.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from webcam.")
            break

        # Mirror display (natural, like a mirror)
        frame_display = cv2.flip(frame, 1)

        # --------------------------------------------------
        # MediaPipe — hand landmark detection & drawing
        # --------------------------------------------------
        rgb_frame = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_detected = False
        right_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        if results.multi_handedness:
            for handedness in results.multi_handedness:
                hand_label = handedness.classification[0].label
                if hand_label == "Left":
                    left_detected = True
                elif hand_label == "Right":
                    right_detected = True

        # --------------------------------------------------
        # Overlay: hand tracking status
        # --------------------------------------------------
        tracking_text = (
            f"Hands: L={'OK' if left_detected else '--'} "
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

        # --------------------------------------------------
        # Overlay: transient message
        # --------------------------------------------------
        if time.time() - transient_message_time < MESSAGE_DURATION:
            cv2.putText(
                frame_display,
                transient_message,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # --------------------------------------------------
        # Recording logic
        # --------------------------------------------------
        if recording:
            # Use the original (non-flipped) frame for feature extraction
            frames_collected = collector.process_frame(frame)

            status_text = f"Recording: {frames_collected} / 30 frames"
            color = (0, 0, 255)  # Red

            if frames_collected >= 30:
                try:
                    filepath = collector.finalize_sequence(category, label)
                    saved_count += 1
                    logger.info("Sequence saved to %s", filepath)

                    transient_message = f"Saved: {os.path.basename(filepath)}"
                    transient_message_time = time.time()
                    recording = False

                    # Console feedback
                    print(f"\n  ✓ Sequence saved: {filepath}")
                    current_count = stats.get_class_counts().get(stats_key, 0)
                    print(f"  Total for '{label}': {current_count}\n")

                except Exception as exc:
                    logger.error("Error saving sequence: %s", exc)
                    transient_message = "Error saving sequence!"
                    transient_message_time = time.time()
                    recording = False

                    # Return True if we saved at least one before the error
                    if saved_count > 0:
                        return True
                    return False
        else:
            status_text = f"Ready: {label}  [r=Record  q=Quit]"
            color = (0, 255, 0)  # Green

        # --------------------------------------------------
        # Overlay: status bar
        # --------------------------------------------------
        cv2.putText(
            frame_display,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # --------------------------------------------------
        # Overlay: live sample count (bottom of frame)
        # --------------------------------------------------
        current_count = stats.get_class_counts().get(stats_key, 0)
        count_text = f"Samples collected: {current_count}"
        cv2.putText(
            frame_display,
            count_text,
            (10, frame_display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # --------------------------------------------------
        # Overlay: label info (bottom-right area)
        # --------------------------------------------------
        label_text = f"{category} / {label}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_pos = frame_display.shape[1] - text_size[0] - 10
        cv2.putText(
            frame_display,
            label_text,
            (x_pos, frame_display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow("FSL Data Collector", frame_display)

        # --------------------------------------------------
        # Keyboard
        # --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Returning to menu...")
            break

        elif key == ord("r") and not recording:
            logger.info("Starting recording for '%s'...", label)
            transient_message = "Recording started!"
            transient_message_time = time.time()
            collector.start_recording()
            recording = True

    return saved_count > 0


# ============================================================
# Main
# ============================================================

def main() -> None:
    # --------------------------------------------------
    # Load label registry
    # --------------------------------------------------
    try:
        registry = LabelRegistry()
    except FileNotFoundError as exc:
        logger.error("Cannot start: %s", exc)
        return
    except ValueError as exc:
        logger.error("Invalid labels.csv: %s", exc)
        return

    logger.info(
        "Loaded %d categories from labels.csv.",
        len(registry.get_categories()),
    )

    # --------------------------------------------------
    # Open shared resources (opened once for entire session)
    # --------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam. Is it connected?")
        return

    stats = CollectionStats()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --------------------------------------------------
    # Session loop
    # --------------------------------------------------
    try:
        with SequenceCollector() as collector:

            category: str | None = None
            label: str | None = None

            while True:

                # ---- Category selection ----
                if category is None:
                    category = select_category(registry)
                    if category is None:
                        logger.info("Exiting.")
                        break
                    label = None  # reset label when category changes

                # ---- Label selection ----
                if label is None:
                    label = select_label(registry, category)
                    if label is None:
                        # User quit label menu — go back to category menu
                        category = None
                        continue

                # ---- Webcam collection ----
                saved = run_collection_session(
                    cap=cap,
                    collector=collector,
                    stats=stats,
                    hands=hands,
                    category=category,
                    label=label,
                )

                # ---- Post-save / post-quit menu ----
                if saved:
                    choice = post_save_menu(label)
                else:
                    # User quit webcam without saving — go back to label menu
                    label = None
                    continue

                if choice is None or choice == 4:
                    logger.info("Exiting.")
                    break
                elif choice == 1:
                    # Record same label again — loop back
                    continue
                elif choice == 2:
                    # Choose another label
                    label = None
                elif choice == 3:
                    # Choose another category
                    category = None
                    label = None

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Session ended.")


if __name__ == "__main__":
    main()

    # python scripts/demo_collection.py

import sys
import os
import cv2
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.collector import SequenceCollector
from src.data_collection.collection_stats import CollectionStats

def main():
    logger.info("========================================")
    logger.info(" FSL Sequence Data Collector (CLI Demo) ")
    logger.info("========================================")
    
    category = input("Enter category (e.g., GREETING): ").strip().upper()
    label = input("Enter label (e.g., GOOD MORNING): ").strip().upper()
    
    if not category or not label:
        logger.error("Category and label cannot be empty.")
        return

    stats = CollectionStats()
    logger.info("\nCurrent collection status:")
    print(stats.get_stats_summary())
    print()

    logger.info(f"Initializing webcam for {label} ({category})...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return
        
    logger.info("Press 'r' to start recording a sequence.")
    logger.info("Press 'q' to quit.")

    # UI State Variables
    transient_message = ""
    transient_message_time = 0
    MESSAGE_DURATION = 2.0  # seconds

    with SequenceCollector() as collector:
        recording = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break
                
            frame_display = frame.copy()
            
            # Display transient messages (like "Sequence Saved!")
            if time.time() - transient_message_time < MESSAGE_DURATION:
                cv2.putText(frame_display, transient_message, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display status
            if recording:
                frames_collected = collector.process_frame(frame)
                status_text = f"Recording: {frames_collected} / 30 frames"
                color = (0, 0, 255) # Red text
                
                # Log frame progress
                if frames_collected > 0:
                    logger.info(f"Captured frame {frames_collected}/30 for {label}")
                
                # If we collected 30 frames, finalize
                if frames_collected >= 30:
                    try:
                        filepath = collector.finalize_sequence(category, label)
                        logger.info(f"Sequence saved successfully to {filepath}")
                        
                        transient_message = f"Saved: seq_{os.path.basename(filepath)}"
                        transient_message_time = time.time()
                        
                        recording = False
                        
                        # Show updated stats
                        logger.info("\nUpdated Stats:")
                        print(stats.get_stats_summary())
                        logger.info("Press 'r' to record another, 'q' to quit.")
                        
                    except Exception as e:
                        logger.error(f"Error saving sequence: {e}")
                        transient_message = "Error saving sequence!"
                        transient_message_time = time.time()
                        recording = False
            else:
                status_text = f"Ready: {label} (Press 'r' to Record, 'q' to Quit)"
                color = (0, 255, 0) # Green text
                
            cv2.putText(frame_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add current sample count to UI
            current_count = stats.get_class_counts().get(f"{category}/{label}", 0)
            count_text = f"Samples collected: {current_count}"
            cv2.putText(frame_display, count_text, (10, frame_display.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Data Collector Demo", frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting collection...")
                break
            elif key == ord('r') and not recording:
                logger.info(f"Starting recording for {label}...")
                transient_message = "Recording Started!"
                transient_message_time = time.time()
                collector.start_recording()
                recording = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

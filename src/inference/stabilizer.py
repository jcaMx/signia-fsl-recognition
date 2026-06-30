from collections import deque, Counter

class PredictionStabilizer:
    def __init__(self, window_size=10, confidence_threshold=0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.predictions = deque(maxlen=window_size)

    def add_prediction(self, class_id, confidence):
        """
        Add a prediction to the stabilizer queue if it meets confidence threshold.
        Returns the stabilized class ID if available, otherwise None.
        """
        if confidence >= self.confidence_threshold:
            self.predictions.append(class_id)
        
        if len(self.predictions) > 0:
            # Return the most common class ID in the window
            return Counter(self.predictions).most_common(1)[0][0]
        return None

    def clear(self):
        self.predictions.clear()

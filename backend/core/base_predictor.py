class BasePredictor:

    mode = "base"

    def predict(self, frame, landmarks):
        raise NotImplementedError
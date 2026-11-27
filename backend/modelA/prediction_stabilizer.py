
"""
HandLex - Model A Prediction Stabilizer

Filters frame-to-frame prediction fluctuations.
Does not modify the trained classifier.
"""

from collections import deque


class PredictionStabilizer:

    def __init__(
        self,
        history_size=7,
        stable_count=5,
        confidence_threshold=0.50,
    ):

        if stable_count > history_size:
            raise ValueError(
                "stable_count cannot exceed history_size"
            )

        self.history = deque(
            maxlen=history_size
        )

        self.stable_count = stable_count
        self.confidence_threshold = confidence_threshold

        self.last_committed = None


    def reset(self):

        self.history.clear()
        self.last_committed = None


    def update(
        self,
        prediction,
        confidence,
        hand_detected=True,
    ):

        # No hand: clear history and allow
        # the same sign to be committed again.
        if not hand_detected:

            self.history.clear()
            self.last_committed = None

            return {
                "stable": False,
                "committed": None,
                "reason": "no_hand",
            }


        # Ignore low-confidence predictions.
        if confidence < self.confidence_threshold:

            self.history.clear()

            return {
                "stable": False,
                "committed": None,
                "reason": "low_confidence",
            }


        # Add current prediction to history.
        self.history.append(prediction)


        # Not enough consistent history yet.
        if len(self.history) < self.stable_count:

            return {
                "stable": False,
                "committed": None,
                "reason": "collecting",
            }


        # Find the most common recent prediction.
        counts = {}

        for sign in self.history:

            counts[sign] = (
                counts.get(sign, 0) + 1
            )


        stable_prediction = max(
            counts,
            key=counts.get
        )


        # Require sufficient agreement.
        if counts[stable_prediction] < self.stable_count:

            return {
                "stable": False,
                "committed": None,
                "reason": "unstable",
            }


        # Prevent repeatedly committing the same sign.
        if stable_prediction == self.last_committed:

            return {
                "stable": True,
                "committed": None,
                "reason": "already_committed",
            }


        self.last_committed = stable_prediction


        return {
            "stable": True,
            "committed": stable_prediction,
            "reason": "committed",
        }

class HandDetector:
    """
    Hand detection interface.

    The actual detector will be implemented
    when we integrate the hand-tracking pipeline.
    """

    def detect(self, frame):
        raise NotImplementedError(
            "Hand detection has not been implemented yet."
        )
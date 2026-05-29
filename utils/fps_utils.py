import time


class FPSCounter:
    def __init__(self):
        self.previous_time = 0

    def get_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.previous_time) if self.previous_time != 0 else 0
        self.previous_time = current_time
        return int(fps)
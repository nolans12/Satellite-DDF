import time


class Timer:
    """Simple verbose timer"""

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.message}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.message} Time: {elapsed_time:.2f}s")

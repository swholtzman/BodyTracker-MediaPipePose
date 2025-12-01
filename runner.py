from __future__ import annotations

import stag  # <--- MUST be first to prevent the crash
import cv2   # <--- Import cv2 second

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'distance_measure'))
from distance_measure.main import Main


class Runner:
    def __init__(self):
        pass

    @staticmethod
    def run():
        print("Running Facial Tracker", file=sys.stderr)
        main = Main()
        print("Distance Measured: ", file=sys.stderr)
        main.run()

if __name__ == '__main__':
    runner = Runner()
    runner.run()
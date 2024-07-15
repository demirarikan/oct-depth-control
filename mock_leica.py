import os
import re

import numpy as np


class MockLeica:
    def __init__(self, scans_path):
        self.scans = self.get_scans_iterator(scans_path)
        self.current_index = 0
        print("Mock Leica initialized")

    def get_scans_iterator(self, scans_path):

        _nsre = re.compile("([0-9]+)")

        def natural_sort_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s)
            ]

        scans = [
            os.path.join(scans_path, scan)
            for scan in os.listdir(scans_path)
            if scan.startswith("volume_") and scan.endswith("npy")
        ]
        scans.sort(key=natural_sort_key)
        return scans

    def __get_b_scans_volume__(self):
        if self.current_index < len(self.scans):
            scan = self.scans[self.current_index]
            self.current_index += 1
            return np.load(scan), None
        else:
            raise StopIteration("No more scans available.")

"""Regression tests for a relay whose final destination overlaps its starting area."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_act import RelayMonitor, FINAL, DESTINATION

class MonitorTests(unittest.TestCase):
    def test_doing_nothing_cannot_pass(self):
        monitor = RelayMonitor()
        for i in range(3000): monitor.update(np.r_[FINAL,.02], {'table'}, i/50)
        self.assertEqual(monitor.events, {})

    def test_receiver_first_cannot_pass(self):
        monitor = RelayMonitor()
        for i in range(20): monitor.update(np.r_[DESTINATION,.06], {'B_fixed','B_moving'}, i/50)
        self.assertEqual(monitor.events, {})

    def test_ordered_sustained_events_pass(self):
        monitor = RelayMonitor(); frame = 0
        for pos, surfaces, length in [
            (np.r_[FINAL,.06], {'A_fixed','A_moving'}, 10),
            (np.r_[DESTINATION,.02], {'table'}, 10),
            (np.r_[DESTINATION,.06], {'B_fixed','B_moving'}, 10),
            (np.r_[FINAL,.02], {'table'}, 25)]:
            for _ in range(length):
                monitor.update(pos, surfaces, frame/50); frame += 1
        self.assertEqual(list(monitor.events), ['A_lift','A_transfer','B_lift','B_final'])

if __name__ == '__main__': unittest.main()

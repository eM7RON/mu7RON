import os
import unittest

import midi

from mu7ron import analyze
from mu7ron.test import _test_variables
from mu7ron.test import _test_parameters
from mu7ron.test import _test_utils


class TestAnalyze(unittest.TestCase):
    @_test_utils.ignore_warnings(ResourceWarning)
    def test_max_simultaneous_notes(self):
        print("\ntesting mu7ron.analyze.max_simultaneous_notes...")

        targets = [1, 1, 1, 0, 1, 2, 3, 4, 5, 6]
        path = os.path.join(os.path.split(__file__)[0], "..", "data", "midi", "test")
        for i, t in zip(range(7, 17), targets):
            sequence = midi.read_midifile(os.path.join(path, f"test{i}.mid"))
            result = analyze.max_simultaneous_notes(sequence)
            self.assertEqual(t, result)


if __name__ == "__main__":
    unittest.main()

import unittest

from mu7ron import maps
from mu7ron.test import _test_variables
from mu7ron.test import _test_parameters


class TestMaps(unittest.TestCase):
    def test_sampling_map(self):
        print("\ntesting mu7ron.maps.sampling_map...")

        t0 = [list(range(5)), list(range(7)), list(range(9))]

        t1 = [list(range(25)), list(range(25)), list(range(99)), list(range(99))]

        t2 = [[]]

        t3 = [
            list(range(7777)),
            list(range(999)),
            list(range(4444)),
            list(range(4444)),
            list(range(7777)),
        ]

        t4 = [
            list(range(7777)),
            list(range(9999)),
            list(range(4444)),
            list(range(4444)),
            list(range(7777)),
        ]

        t5 = []

        v = _test_variables.v40()

        self.assertEqual(maps.sampling_map(t0, 3, 1), v["t0"])
        self.assertEqual(maps.sampling_map(t1, 13, 7), v["t1"])
        self.assertEqual(maps.sampling_map(t2, 13, 7), v["t2"])
        self.assertEqual(maps.sampling_map(t3, 777, 777), v["t3"])
        self.assertEqual(maps.sampling_map(t4, 777, 4444), v["t4"])
        self.assertEqual(maps.sampling_map(t5, 1, 1), v["t5"])
        self.assertEqual(maps.sampling_map(t5, 0, 1), v["t5"])


if __name__ == "__main__":
    unittest.main()

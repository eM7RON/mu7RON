import unittest

import numpy as np

from mu7ron import aug
from mu7ron.test import _test_variables
from mu7ron.test import _test_parameters

class TestAug(unittest.TestCase):
    def test_translate(self):
        print('\ntesting mu7ron.aug.transpose...')
        # Test if returns invalid sequence based on parameters
        for _ in range(100):
            off_mode = bool(np.random.randint(2))
            n_time   = np.random.randint(256)
            n_pitch  = 256 if off_mode else 128
            n_t_p    = n_time + n_pitch
            for i in range(0, 64, 8):
                input_sequence = np.random.randint(n_time + i, n_t_p - i, size=np.random.randint(1, 1000))
                for j in range(-64, 64, 8):
                    result = aug.transpose(input_sequence, j, n_time=n_time, n_t_p=n_t_p, off_mode=off_mode)
                    self.assertTrue(bool(len(result)))
                    self.assertTrue(max(result) < n_t_p)
                    self.assertTrue(min(result) >= n_time)
        # Test if returns correct sequence
        p = _test_parameters.p0()
        v = _test_variables.v41()
        t = np.array([4, 78, 100, 256, 305])
        n_t_p = p['n_time'] + p['n_pitch']
        result = aug.transpose(t, -7, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t0'], result))
        result = aug.transpose(t, 13, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t1'], result))
        
        t = np.array([*[47] * 100, 256, *[304] * 100])
        result = aug.transpose(t, -7, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t2'], result))
        result = aug.transpose(t, 13, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t3'], result))
        
        t = np.array([])
        result = aug.transpose(t, -7, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t4'], result))
        t = np.array([*[777] * 100])
        result = aug.transpose(t, 13, n_time=p['n_time'], n_t_p=n_t_p, off_mode=p['off_mode'])
        self.assertTrue(np.array_equal(v['t5'], result))


if __name__ == '__main__':
    unittest.main()
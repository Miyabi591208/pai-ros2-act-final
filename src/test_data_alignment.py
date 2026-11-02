import unittest
import numpy as np
from data_alignment import JOINTS, reorder, causal_sample, uniform_times, action_offsets, action_chunk


class AlignmentTests(unittest.TestCase):
    def test_shuffled_names(self):
        order = [4, 2, 0, 3, 1]
        np.testing.assert_equal(reorder([JOINTS[i] for i in order], order), np.arange(5))

    def test_missing_joint_is_not_zero(self):
        with self.assertRaises(ValueError):
            reorder(JOINTS[:-1], [0]*4)

    def test_no_future_and_stale_data(self):
        with self.assertRaises(ValueError):
            causal_sample([100, 200], [1, 2], [99])
        with self.assertRaises(ValueError):
            causal_sample([100, 200], [1, 2], [300], max_age_ns=50)
        np.testing.assert_equal(causal_sample([100, 200], [1, 2], [199,200])[0], [1,2])

    def test_fixed_period(self):
        t = uniform_times(1_700_000_000_000_000_000, 1_700_000_001_030_000_000)
        np.testing.assert_equal(np.diff(t), 50_000_000)

    def test_invalid_frequency(self):
        for fps in [0,-1,3]:
            with self.assertRaises(ValueError): uniform_times(0,1_000_000_000,fps)

    def test_action_zero_is_current(self):
        self.assertEqual(action_offsets()[0], 0)
        np.testing.assert_allclose(np.diff(action_offsets()), .05)

    def test_end_mask(self):
        data, mask = action_chunk(np.arange(5)[:,None],3,4)
        np.testing.assert_equal(data[:,0], [3,4,4,4])
        np.testing.assert_equal(mask, [False,False,True,True])


if __name__ == '__main__':
    unittest.main()

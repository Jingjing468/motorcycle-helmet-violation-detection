import unittest

from src.utils.box_utils import box_iou, box_center, clip_box


class BoxUtilsTests(unittest.TestCase):
    def test_iou_for_overlapping_boxes(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)

    def test_center_and_clipping(self):
        self.assertEqual(box_center((0, 2, 8, 10)), (4, 6))
        self.assertEqual(clip_box((-2, 3, 12, 20), 10, 15), (0, 3, 10, 15))


if __name__ == "__main__":
    unittest.main()

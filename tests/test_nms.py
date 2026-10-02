import unittest

from src.postprocess.nms import class_aware_nms


class NmsTests(unittest.TestCase):
    def test_suppresses_overlapping_boxes_per_class(self):
        items = [
            {"class_name": "bike", "confidence": .9, "xyxy": (0, 0, 20, 20)},
            {"class_name": "bike", "confidence": .7, "xyxy": (1, 1, 19, 19)},
            {"class_name": "helmet", "confidence": .8, "xyxy": (1, 1, 19, 19)},
        ]
        self.assertEqual(len(class_aware_nms(items, .5)), 2)


if __name__ == "__main__":
    unittest.main()

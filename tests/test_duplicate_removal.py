import unittest

from src.postprocess.duplicate_removal import motorcycle_boxes_duplicate


class DuplicateRemovalTests(unittest.TestCase):
    def test_overlapping_bike_predictions_are_duplicates(self):
        first = {"xyxy": (0, 0, 100, 100)}
        second = {"xyxy": (5, 5, 95, 95)}
        self.assertTrue(motorcycle_boxes_duplicate(first, second))

    def test_separate_bikes_are_not_duplicates(self):
        self.assertFalse(motorcycle_boxes_duplicate(
            {"xyxy": (0, 0, 50, 50)}, {"xyxy": (100, 0, 150, 50)}))


if __name__ == "__main__":
    unittest.main()

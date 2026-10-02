import unittest

from src.visualization.detection_summary import render_detection_summary


class DetectionSummaryTests(unittest.TestCase):
    def test_counts_rider_rows_and_confidence_word_are_present(self):
        bikes = [{}, {}]
        by_bike = {0: [{"class_name": "no-helmet", "confidence": .90}], 1: []}
        html = render_detection_summary(bikes, by_bike, 0, 1, 0)
        self.assertIn("Motorcycles", html)
        self.assertIn("Rider Status", html)
        self.assertIn("Bike 1", html)
        self.assertIn("90% confidence", html)
        self.assertIn("No Helmet", html)
        self.assertIn("Undetermined", html)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.ocr.plate_parser import normalize_plate_text, registration_candidate


class PlateParserTests(unittest.TestCase):
    def test_normalizes_ascii_plate_characters(self):
        self.assertEqual(normalize_plate_text("ab 12--345"), "AB 12-345".replace(" ", ""))

    def test_accepts_plate_like_registration_and_rejects_short_text(self):
        self.assertEqual(registration_candidate("1A-2345"), "1A-2345")
        self.assertIsNone(registration_candidate("A12"))


if __name__ == "__main__":
    unittest.main()

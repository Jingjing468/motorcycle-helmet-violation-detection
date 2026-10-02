import unittest

from src.association.rider_bike_association import associate_people_to_bikes


class RiderAssociationTests(unittest.TestCase):
    def test_person_associates_to_nearest_plausible_bike(self):
        persons = [{"xyxy": (30, 0, 60, 50), "confidence": .9}]
        bikes = [{"xyxy": (0, 30, 80, 100)}, {"xyxy": (200, 30, 280, 100)}]
        associations = associate_people_to_bikes(persons, bikes)
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0][2], 0)


if __name__ == "__main__":
    unittest.main()

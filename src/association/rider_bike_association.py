"""Assign each detected person to the nearest spatially plausible motorcycle."""
from src.association.spatial_matching import person_bike_distance


def nearest_bike(person, bikes):
    matches = [(distance, index) for index, bike in enumerate(bikes)
               if (distance := person_bike_distance(person["xyxy"], bike["xyxy"])) is not None]
    return min(matches) if matches else None


def associate_people_to_bikes(persons, bikes):
    associations = []
    for person_index, person in enumerate(persons):
        match = nearest_bike(person, bikes)
        if match is not None:
            distance, bike_index = match
            associations.append((person_index, person, bike_index, distance))
    return associations

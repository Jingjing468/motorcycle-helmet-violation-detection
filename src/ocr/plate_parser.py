"""OCR text normalization following the existing plate candidate rules."""
import re


def normalize_plate_text(text):
    cleaned = re.sub(r"[^A-Z0-9-]", "", str(text).upper())
    return re.sub(r"-+", "-", cleaned).strip("-")


def registration_candidate(text):
    cleaned = normalize_plate_text(text)
    digits = sum(char.isdigit() for char in cleaned)
    letters = sum(char.isalpha() for char in cleaned)
    return cleaned if 5 <= len(cleaned) <= 12 and digits >= 3 and 1 <= letters <= 5 else None


def combine_lines(lines):
    return "".join(normalize_plate_text(line) for line in lines if normalize_plate_text(line))

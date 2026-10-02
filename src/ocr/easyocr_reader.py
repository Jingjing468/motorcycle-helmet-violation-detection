"""Thin wrapper around EasyOCR that isolates the optional dependency."""
def read_plate(reader, image, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"):
    if reader is None:
        return []
    return reader.readtext(image, detail=1, paragraph=False, allowlist=allowlist)

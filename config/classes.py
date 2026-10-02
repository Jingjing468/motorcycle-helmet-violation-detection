"""Custom detector class IDs and readable labels."""
CLASS_NAMES = {0: "bike", 1: "helmet", 2: "no-helmet", 3: "number-plate"}
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
HEAD_CLASSES = frozenset(("helmet", "no-helmet"))

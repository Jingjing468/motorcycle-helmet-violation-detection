"""Typed result records shared between the pipeline and reporting code."""
from dataclasses import dataclass, field
from typing import Any

Box = tuple[float, float, float, float]


@dataclass
class Detection:
    class_name: str
    confidence: float
    xyxy: Box
    source: str = ""


@dataclass
class BikeResult:
    index: int
    detection: Detection
    riders: list["RiderResult"] = field(default_factory=list)


@dataclass
class RiderResult:
    bike_index: int
    status: str
    confidence: float | None


@dataclass
class PlateResult:
    detection: Detection
    text: str = "Unreadable"


@dataclass
class PipelineResult:
    image: Any
    violation: bool
    bikes: list[BikeResult] = field(default_factory=list)
    plates: list[PlateResult] = field(default_factory=list)

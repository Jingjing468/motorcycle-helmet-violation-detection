"""Gradio entry point for motorcycle helmet violation detection."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_loader import load_models
from src.core.pipeline import run_pipeline
from app.ui_components import build_app
from app.ui_styles import APP_CSS

model, coco_model, reader = load_models()


def web_detect(image):
    """Keep the original Gradio callback contract while delegating inference."""
    return run_pipeline(image, model, coco_model, reader)


app = build_app(web_detect)


if __name__ == "__main__":
    app.launch(css=APP_CSS)

import importlib
from pathlib import Path

import gradio as gr  # type: ignore[import-not-found]
import cv2  # type: ignore[import-not-found]

try:
    easyocr = importlib.import_module("easyocr")
except ImportError:  # pragma: no cover - optional dependency for OCR path
    easyocr = None

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for model path
    YOLO = None

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "helmet_best.pt"

# Load Deep Learning model
model = YOLO(str(MODEL_PATH)) if YOLO is not None else None

# Load OCR
reader = easyocr.Reader(["en"]) if easyocr is not None else None


def web_detect(image):
    if image is None:
        return None, "Please upload an image", "Not detected", ""

    if model is None:
        return None, "Model dependency is missing. Please install ultralytics.", "Not detected", ""

    if reader is None:
        return None, "OCR dependency is missing. Please install easyocr.", "Not detected", ""

    temp_path = PROJECT_ROOT / "temp_input.jpg"

    cv2.imwrite(
        str(temp_path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    # YOLO prediction
    results = model.predict(
        source=str(temp_path),
        conf=0.25
    )

    violation = False
    plate_text = "Not detected"
    detection_lines = []

    original = cv2.imread(str(temp_path))

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            detection_lines.append(
                f"{class_name}: {confidence * 100:.1f}%"
            )

            # Helmet violation
            if class_name == "no-helmet":
                violation = True

            # Number plate
            if class_name == "number-plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate = original[y1:y2, x1:x2]

                if plate.size > 0:
                    # Enlarge plate
                    plate = cv2.resize(
                        plate,
                        None,
                        fx=4,
                        fy=4,
                        interpolation=cv2.INTER_CUBIC
                    )

                    # Improve OCR image
                    gray = cv2.cvtColor(
                        plate,
                        cv2.COLOR_BGR2GRAY
                    )

                    gray = cv2.equalizeHist(gray)

                    plate_path = PROJECT_ROOT / "temp_plate.jpg"
                    cv2.imwrite(str(plate_path), gray)

                    # OCR
                    ocr_result = reader.readtext(str(plate_path))

                    texts = []

                    for detection in ocr_result:
                        text = detection[1]
                        ocr_confidence = detection[2]

                        if ocr_confidence >= 0.15:
                            texts.append(text)

                    if texts:
                        plate_text = " ".join(texts)

    # Draw bounding boxes
    detected_image = results[0].plot()

    detected_image = cv2.cvtColor(
        detected_image,
        cv2.COLOR_BGR2RGB
    )

    if violation:
        status = "🚨 HELMET VIOLATION DETECTED"
    else:
        status = "✅ NO HELMET VIOLATION"

    detection_summary = "\n".join(detection_lines)

    return (
        detected_image,
        status,
        plate_text,
        detection_summary
    )


# Web Interface
with gr.Blocks(
    title="Motorcycle Helmet Violation Detection"
) as app:

    gr.Markdown(
        """
        # 🏍️ Motorcycle Helmet Violation Detection

        Deep Learning-based traffic safety system using **YOLOv8 + EasyOCR**.

        Upload a motorcycle traffic image to detect helmets,
        no-helmet violations, motorcycles, and license plates.
        """
    )

    with gr.Row():

        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                label="Upload Traffic Image"
            )

            analyze_button = gr.Button(
                "Analyze Image",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                value="Clear"
            )

        with gr.Column():
            result_image = gr.Image(
                label="Detection Result"
            )

    gr.Markdown("## Detection Summary")

    with gr.Row():

        status_box = gr.Textbox(
            label="Violation Status"
        )

        plate_box = gr.Textbox(
            label="License Plate"
        )

    detection_box = gr.Textbox(
        label="Detected Objects & Confidence",
        lines=6
    )

    analyze_button.click(
        fn=web_detect,
        inputs=input_image,
        outputs=[
            result_image,
            status_box,
            plate_box,
            detection_box
        ]
    )

    clear_button.add([
        input_image,
        result_image,
        status_box,
        plate_box,
        detection_box
    ])


if __name__ == "__main__":
    app.launch()
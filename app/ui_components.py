"""Composable Gradio page sections and control wiring."""
import gradio as gr  # type: ignore[import-not-found]


def build_app(analyze_callback):
    """Build the upload, status, plate, and detection-summary interface."""
    with gr.Blocks(title="Motorcycle Helmet Violation Detection") as app:
        gr.Markdown("""
        # 🏍️ Motorcycle Helmet Violation Detection

        Deep Learning-based traffic safety system using **YOLOv8 + EasyOCR**.

        Upload a motorcycle traffic image to detect helmets, no-helmet violations, motorcycles, and license plates.
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Traffic Image")
                analyze_button = gr.Button("Analyze Image", variant="primary")
                clear_button = gr.ClearButton(value="Clear")
            with gr.Column():
                result_image = gr.Image(label="Detection Result")
        gr.Markdown("## Detection Summary")
        with gr.Row():
            status_box = gr.Textbox(label="Violation Status")
            plate_box = gr.Textbox(label="Detected License Plates", lines=8)
        detection_box = gr.HTML()
        outputs = [result_image, status_box, plate_box, detection_box]
        analyze_button.click(fn=analyze_callback, inputs=input_image, outputs=outputs)
        clear_button.add([input_image, *outputs])
    return app

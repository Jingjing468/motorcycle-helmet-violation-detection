"""Composable Gradio page sections and control wiring."""
import gradio as gr  # type: ignore[import-not-found]


def build_app(analyze_callback):
    """Build the upload, result, status, plate, and summary interface."""
    with gr.Blocks(title="Motorcycle Helmet Violation Detection") as app:
        with gr.Row():
            with gr.Column(scale=5, elem_classes="app-header"):
                gr.Markdown(
                    "# Motorcycle Helmet Violation Detection\n"
                    "AI-powered traffic safety analysis using YOLOv8"
                )
            with gr.Column(scale=1, min_width=150, elem_classes="model-badge"):
                gr.Markdown("🟢 Model Ready<br><small>YOLOv8</small>")

        with gr.Row(equal_height=True, elem_classes="analysis-row"):
            with gr.Column(scale=1, elem_classes="analysis-panel"):
                gr.Markdown("## Input Image")
                input_image = gr.Image(type="numpy", label="Upload Traffic Image", elem_classes="analysis-image")
                with gr.Row(elem_classes="action-row"):
                    analyze_button = gr.Button("Analyze Image", variant="primary")
                    clear_button = gr.ClearButton(value="Clear")
            with gr.Column(scale=1, elem_classes="analysis-panel"):
                gr.Markdown("## Detection Result")
                result_image = gr.Image(label="Annotated detections", elem_classes="analysis-image")
                object_gallery = gr.Gallery(
                    label="Detected Objects", columns=4, rows=1, height=190,
                    object_fit="contain", elem_classes="object-gallery",
                    elem_id="detected-objects-gallery",
                )

        with gr.Row(elem_classes="info-row"):
            with gr.Column(scale=1, elem_classes="info-panel"):
                status_box = gr.Textbox(label="Helmet Violation Status", value="Waiting for analysis", lines=2, interactive=False, elem_id="violation-status")
            with gr.Column(scale=1, elem_classes="info-panel"):
                plate_box = gr.Textbox(label="License Plate Result", placeholder="Waiting for analysis", lines=4, interactive=False, elem_id="plate-result")

        gr.Markdown("## Detection Summary", elem_classes="section-title")
        detection_box = gr.HTML()
        gr.HTML("""<footer class='technical-footer'>YOLOv8&nbsp;&nbsp; · &nbsp;&nbsp;640 × 640&nbsp;&nbsp; · &nbsp;&nbsp;PyTorch + Ultralytics&nbsp;&nbsp; · &nbsp;&nbsp;EasyOCR&nbsp;&nbsp; · &nbsp;&nbsp;40 Epochs</footer>""")

        outputs = [result_image, status_box, plate_box, detection_box, object_gallery]
        analyze_button.click(fn=analyze_callback, inputs=input_image, outputs=outputs)
        clear_button.add([input_image, *outputs])
    return app

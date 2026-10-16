"""Scoped styling for the Gradio demonstration interface."""
APP_CSS = r"""
/* Limit page-level styling to this app's existing header scope. */
.gradio-container:has(.app-header) { max-width: 1500px !important; padding: 24px !important; background: #0D0F0E !important; color: #F5F7F6 !important; }
.app-header h1, .app-header h2, .app-header h3,
.analysis-panel h1, .analysis-panel h2, .analysis-panel h3,
.info-panel h1, .info-panel h2, .info-panel h3,
.section-title h1, .section-title h2, .section-title h3 { color: #F5F7F6 !important; }
.app-header { padding: 4px 0 14px !important; }
.app-header h1 { margin-bottom: 5px !important; font-size: clamp(25px, 3vw, 34px) !important; }
.app-header p { color: #9CA3AF !important; }
.model-badge { align-self: center; padding: 10px 14px !important; border: 1px solid #2A302D !important; border-radius: 10px !important; background: #151817 !important; }
.model-badge p { margin: 0 !important; color: #F5F7F6 !important; font-size: 13px !important; }
.analysis-row { gap: 18px !important; align-items: stretch !important; }
.analysis-panel { min-width: 0 !important; padding: 16px !important; border: 1px solid #2A302D !important; border-radius: 15px !important; background: #151817 !important; color: #F5F7F6 !important; }
.analysis-panel h2 { margin: 0 0 10px !important; font-size: 18px !important; }
.analysis-image { height: 520px !important; color: #F5F7F6 !important; }
.analysis-image > div { height: 520px !important; min-height: 520px !important; border: 1px solid #2A302D !important; border-radius: 11px !important; background: #1B1F1D !important; color: #F5F7F6 !important; }
/* Gradio nests white upload and preview surfaces inside the image component. */
.analysis-image div, .analysis-image [data-testid] {
 background-color: #1B1F1D !important; color: #F5F7F6 !important;
 border-color: #2A302D !important;
}
.analysis-image img { width: 100% !important; height: 100% !important; object-fit: contain !important; background: transparent !important; }
.action-row { margin-top: 12px !important; gap: 10px !important; }
.action-row button { min-height: 42px !important; border-radius: 9px !important; }
.action-row button.primary { background: #16A34A !important; border-color: #16A34A !important; color: #fff !important; }
.action-row button.primary:hover { background: #15803D !important; border-color: #15803D !important; }
.action-row button:not(.primary) { background: #1B1F1D !important; border: 1px solid #343A37 !important; color: #F5F7F6 !important; }
.info-row { gap: 16px !important; margin-top: 20px !important; }
.info-panel { padding: 16px !important; border: 1px solid #2A302D !important; border-radius: 12px !important; background: #151817 !important; color: #F5F7F6 !important; }
.info-panel label { color: #F5F7F6 !important; }
.info-panel textarea, .info-panel input {
 background: #1B1F1D !important; border: 1px solid #2A302D !important;
 border-radius: 10px !important; color: #F5F7F6 !important;
 -webkit-text-fill-color: #F5F7F6 !important;
}
.info-panel textarea::placeholder, .info-panel input::placeholder { color: #9CA3AF !important; opacity: 1 !important; }
.info-panel > div, .info-panel .wrap, .info-panel .container,
.info-panel [data-testid="textbox"] {
 background: #151817 !important; border-color: #2A302D !important; border-radius: 10px !important;
}
.info-panel [data-testid="textbox"] textarea { background: #1B1F1D !important; }
.section-title { margin: 22px 0 8px !important; }
.summary-grid { gap: 10px !important; }
.summary-card { padding: 13px 15px !important; border-color: #2A302D !important; border-radius: 11px !important; background: #151817 !important; }
.summary-card span { color: #9CA3AF !important; }
.summary-card strong { color: #F5F7F6 !important; }
.summary-card.no-helmet-count strong { color: #EF4444 !important; }
.summary-card.helmet-count strong { color: #22C55E !important; }
.rider-card { border-color: #2A302D !important; border-radius: 11px !important; background: #151817 !important; }
.rider-card h3 { color: #F5F7F6 !important; border-color: #2A302D !important; }
.rider-card table, .rider-card td { color: #E5E9E7 !important; }
.rider-card th { color: #9CA3AF !important; background: #1B1F1D !important; border-color: #2A302D !important; }
.rider-card td { border-color: #242A27 !important; }
.no-helmet-status { color: #F87171 !important; }
.helmet-status { color: #4ADE80 !important; }
.unknown-status, .empty-state { color: #9CA3AF !important; }
.object-gallery { margin-top: 16px !important; }
.object-gallery label { color: #F5F7F6 !important; font-weight: 650 !important; }
.object-gallery [data-testid="gallery"] { background: #151817 !important; border: 1px solid #2A302D !important; border-radius: 12px !important; color: #F5F7F6 !important; }
.object-gallery > div { background: #151817 !important; color: #F5F7F6 !important; }
.object-gallery [data-testid="gallery"] div { background: #1B1F1D !important; color: #F5F7F6 !important; }
.object-gallery [data-testid="gallery"] figure,
.object-gallery [data-testid="gallery"] [class*="thumbnail"],
.object-gallery [data-testid="gallery"] [class*="gallery-item"],
.object-gallery [data-testid="gallery"] [class*="grid-item"] {
 background: #1B1F1D !important; border: 1px solid #2A302D !important;
 border-radius: 11px !important; color: #F5F7F6 !important;
}
.object-gallery [data-testid="gallery"] figcaption,
.object-gallery [data-testid="gallery"] [class*="caption"] { background: #111412 !important; color: #F5F7F6 !important; }
.object-gallery [data-testid="gallery"] figcaption small,
.object-gallery [data-testid="gallery"] [class*="caption"] small { color: #9CA3AF !important; }
.object-gallery [data-testid="gallery"] img { background: #1B1F1D !important; border-radius: 8px !important; }
.object-gallery [data-testid="gallery"] button { color: #F5F7F6 !important; }
/* Native Gradio component wrappers, scoped to the three result components. */
#detected-objects-gallery, #violation-status, #plate-result {
 --background-fill-primary: #151817 !important;
 --background-fill-secondary: #1B1F1D !important;
 --block-background-fill: #151817 !important;
 --input-background-fill: #1B1F1D !important;
 --block-border-color: #2A302D !important;
 --body-text-color: #F5F7F6 !important;
 background: #151817 !important;
 color: #F5F7F6 !important;
 border-color: #2A302D !important;
}
#detected-objects-gallery > div, #detected-objects-gallery div,
#detected-objects-gallery .wrap, #detected-objects-gallery .container,
#detected-objects-gallery .gallery, #detected-objects-gallery .gallery-container,
#detected-objects-gallery .grid-wrap, #detected-objects-gallery .grid-container {
 background: #151817 !important; color: #F5F7F6 !important;
 border-color: #2A302D !important;
}
#detected-objects-gallery .thumbnail-item,
#detected-objects-gallery .gallery-item,
#detected-objects-gallery figure, #detected-objects-gallery label {
 background: #1B1F1D !important; color: #F5F7F6 !important;
 border-color: #2A302D !important; border-radius: 11px !important;
}
#detected-objects-gallery figcaption,
#detected-objects-gallery [class*="caption"] {
 background: #111412 !important; color: #F5F7F6 !important;
}
#detected-objects-gallery span, #detected-objects-gallery p,
#violation-status span, #violation-status p,
#plate-result span, #plate-result p { color: #F5F7F6 !important; }
#violation-status > div, #violation-status div, #violation-status .wrap,
#violation-status .form, #violation-status label,
#violation-status [data-testid="textbox"],
#plate-result > div, #plate-result div, #plate-result .wrap,
#plate-result .form, #plate-result label,
#plate-result [data-testid="textbox"] {
 background: #151817 !important; color: #F5F7F6 !important;
 border-color: #2A302D !important;
}
#violation-status textarea, #violation-status input,
#plate-result textarea, #plate-result input {
 background: #1B1F1D !important; color: #F5F7F6 !important;
 -webkit-text-fill-color: #F5F7F6 !important;
 border: 1px solid #2A302D !important;
}
#violation-status textarea::placeholder, #plate-result textarea::placeholder {
 color: #9CA3AF !important; opacity: 1 !important;
}
.technical-footer { margin-top: 25px; padding: 14px 2px 0; border-top: 1px solid #252B28; color: #858D88; font-size: 11px; }
@media (max-width: 760px) {
 .gradio-container:has(.app-header) { padding: 14px !important; }
 .analysis-row, .info-row { flex-direction: column !important; }
 .analysis-image, .analysis-image > div { height: min(70vw, 500px) !important; min-height: min(70vw, 500px) !important; }
}
"""

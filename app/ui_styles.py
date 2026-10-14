"""Scoped styling for the Gradio demonstration interface."""
APP_CSS = r"""
.gradio-container { max-width: 1500px !important; padding: 24px !important; background: #0D0F0E !important; color: #F5F7F6 !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: #F5F7F6 !important; }
.app-header { padding: 4px 0 14px !important; }
.app-header h1 { margin-bottom: 5px !important; font-size: clamp(25px, 3vw, 34px) !important; }
.app-header p { color: #9CA3AF !important; }
.model-badge { align-self: center; padding: 10px 14px !important; border: 1px solid #2A302D !important; border-radius: 10px !important; background: #151817 !important; }
.model-badge p { margin: 0 !important; color: #F5F7F6 !important; font-size: 13px !important; }
.analysis-row { gap: 18px !important; align-items: stretch !important; }
.analysis-panel { min-width: 0 !important; padding: 16px !important; border: 1px solid #2A302D !important; border-radius: 14px !important; background: #151817 !important; }
.analysis-panel h2 { margin: 0 0 10px !important; font-size: 18px !important; }
.analysis-image { height: 520px !important; }
.analysis-image > div { height: 520px !important; min-height: 520px !important; border: 1px solid #39413D !important; border-radius: 10px !important; background: #1B1F1D !important; }
.analysis-image img { width: 100% !important; height: 100% !important; object-fit: contain !important; }
.action-row { margin-top: 12px !important; gap: 10px !important; }
.action-row button { min-height: 42px !important; border-radius: 9px !important; }
.action-row button.primary { background: #16A34A !important; border-color: #16A34A !important; color: #fff !important; }
.action-row button.primary:hover { background: #15803D !important; border-color: #15803D !important; }
.action-row button:not(.primary) { background: #1B1F1D !important; border: 1px solid #39413D !important; color: #E5E9E7 !important; }
.info-row { gap: 16px !important; margin-top: 20px !important; }
.info-panel { padding: 16px !important; border: 1px solid #2A302D !important; border-radius: 12px !important; background: #151817 !important; }
.info-panel textarea { background: #1B1F1D !important; border-color: #39413D !important; color: #F5F7F6 !important; }
.info-panel label { color: #D1D5D3 !important; }
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
.object-gallery [data-testid="gallery"] { background: #151817 !important; border-color: #2A302D !important; }
.technical-footer { margin-top: 25px; padding: 14px 2px 0; border-top: 1px solid #252B28; color: #858D88; font-size: 11px; }
@media (max-width: 760px) {
 .gradio-container { padding: 14px !important; }
 .analysis-row, .info-row { flex-direction: column !important; }
 .analysis-image, .analysis-image > div { height: min(70vw, 500px) !important; min-height: min(70vw, 500px) !important; }
}
"""

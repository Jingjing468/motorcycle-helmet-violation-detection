"""HTML summary for the validated detection results."""
from html import escape


SUMMARY_CSS = """
.detection-summary { color: #F5F7F6; font-family: inherit; }
.summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.summary-card { box-sizing: border-box; min-width: 0; padding: 16px 18px; border: 1px solid #2A302D; border-radius: 10px; background: #151817; }
.summary-card span { display: block; color: #9CA3AF; font-size: 14px; }
.summary-card strong { display: block; margin-top: 6px; color: #F5F7F6; font-size: 25px; line-height: 1.2; }
.summary-card.no-helmet-count strong, .no-helmet-status { color: #EF4444; }
.summary-card.helmet-count strong, .helmet-status { color: #22C55E; }
.no-helmet-status { font-weight: 600; }
.rider-card { margin-top: 18px; overflow: hidden; border: 1px solid #2A302D; border-radius: 8px; background: #151817; }
.rider-card h3 { margin: 0; padding: 15px 18px; color: #F5F7F6; border-bottom: 1px solid #2A302D; font-size: 16px; font-weight: 600; }
.rider-card table { width: 100%; border-collapse: collapse; text-align: left; color: #E5E9E7; }
.rider-card th, .rider-card td { padding: 12px 18px; border-bottom: 1px solid #242A27; }
.rider-card th { color: #9CA3AF; background: #1B1F1D; font-size: 12px; font-weight: 600; }
.rider-card tr:last-child td { border-bottom: 0; }
.unknown-status, .empty-state { color: #9CA3AF; }
@media (max-width: 640px) { .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; } .summary-card { padding: 13px; } .rider-card th, .rider-card td { padding: 10px 12px; } }
"""


def render_detection_summary(bikes, validated_by_bike, helmet_count, no_helmet_count, plate_count):
    """Render count cards and one status row per validated rider."""
    cards = (("Motorcycles", len(bikes)), ("Helmet", helmet_count),
             ("No Helmet", no_helmet_count), ("License Plates", plate_count))
    card_html = "".join(
        f'<div class="summary-card {"no-helmet-count" if label == "No Helmet" else "helmet-count" if label == "Helmet" and count > 0 else ""}">'
        f"<span>{label}</span><strong>{count}</strong></div>"
        for label, count in cards
    )
    rows = []
    for index, _bike in enumerate(bikes, start=1):
        occupants = validated_by_bike[index - 1]
        if not occupants:
            rows.append(f'<tr><td>Bike {index}</td><td class="unknown-status">Undetermined</td><td>—</td></tr>')
        for head in occupants:
            name = escape(head["class_name"].replace("-", " ").title())
            css = "no-helmet-status" if head["class_name"] == "no-helmet" else "helmet-status"
            rows.append(
                f'<tr><td>Bike {index}</td><td class="{css}">{name}</td>'
                f'<td>{head["confidence"] * 100:.0f}% confidence</td></tr>'
            )
    if not rows:
        rows.append('<tr><td colspan="3" class="empty-state">No motorcycles detected.</td></tr>')
    return f'''<section class="detection-summary" aria-label="Detection Summary">
      <style>{SUMMARY_CSS}</style><div class="summary-grid">{card_html}</div>
      <div class="rider-card"><h3>Rider Status</h3><table>
      <thead><tr><th>Motorcycle</th><th>Status</th><th>Confidence</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table></div></section>'''

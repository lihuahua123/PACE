#!/usr/bin/env python3
"""Render SVG and PNG charts for Mini SWE Agent intent statistics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import struct
import zlib
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "/workspace/skill/analysis/outputs/minisweagent_intent_classification/"
    "batch-core12-swebp-20260709_160007-DeepSeek-V4-Flash/summary.json"
)
TOP_N = 12
CATEGORY_ORDER = [
    "read",
    "search",
    "edit",
    "verify",
    "git",
    "housekeeping",
    "failed",
    "other",
]
CATEGORY_COLORS = {
    "read": "#5a7d9a",
    "search": "#7c9b6f",
    "edit": "#4a8a5a",
    "verify": "#b56a50",
    "git": "#3a8a8a",
    "housekeeping": "#6f8a8d",
    "failed": "#a05050",
    "other": "#888888",
}
TOP_BAR_COLOR = "#5a7d9a"
BG = "#fcfbf7"
TRACK = "#ece7dc"
TEXT = "#222222"
SUBTEXT = "#666666"
LABEL = "#444444"
WHITE = "#ffffff"

# 5x7 bitmap font; each row uses 5 bits.
FONT = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 31, 0, 0, 0],
    ".": [0, 0, 0, 0, 0, 12, 12],
    ":": [0, 12, 12, 0, 12, 12, 0],
    "%": [17, 2, 4, 8, 17, 0, 0],
    "(": [2, 4, 8, 8, 8, 4, 2],
    ")": [8, 4, 2, 2, 2, 4, 8],
    "/": [1, 2, 4, 8, 16, 0, 0],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31],
    "3": [30, 1, 1, 14, 1, 1, 30],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 16, 30, 1, 1, 30],
    "6": [14, 16, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 1, 14],
    "a": [0, 0, 14, 1, 15, 17, 15],
    "b": [16, 16, 22, 25, 17, 17, 30],
    "c": [0, 0, 14, 16, 16, 17, 14],
    "d": [1, 1, 13, 19, 17, 17, 15],
    "e": [0, 0, 14, 17, 31, 16, 14],
    "f": [6, 8, 8, 28, 8, 8, 8],
    "g": [0, 0, 15, 17, 15, 1, 14],
    "h": [16, 16, 22, 25, 17, 17, 17],
    "i": [4, 0, 12, 4, 4, 4, 14],
    "j": [2, 0, 6, 2, 2, 18, 12],
    "k": [16, 16, 18, 20, 24, 20, 18],
    "l": [12, 4, 4, 4, 4, 4, 14],
    "m": [0, 0, 26, 21, 21, 21, 21],
    "n": [0, 0, 22, 25, 17, 17, 17],
    "o": [0, 0, 14, 17, 17, 17, 14],
    "p": [0, 0, 30, 17, 30, 16, 16],
    "q": [0, 0, 13, 19, 15, 1, 1],
    "r": [0, 0, 22, 25, 16, 16, 16],
    "s": [0, 0, 15, 16, 14, 1, 30],
    "t": [8, 8, 28, 8, 8, 8, 6],
    "u": [0, 0, 17, 17, 17, 19, 13],
    "v": [0, 0, 17, 17, 17, 10, 4],
    "w": [0, 0, 17, 17, 21, 21, 10],
    "x": [0, 0, 17, 10, 4, 10, 17],
    "y": [0, 0, 17, 17, 15, 1, 14],
    "z": [0, 0, 31, 2, 4, 8, 31],
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 15],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [14, 4, 4, 4, 4, 4, 14],
    "J": [7, 2, 2, 2, 2, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 17, 17, 17, 17],
    "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 17, 21, 27, 17],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
    "Z": [31, 1, 2, 4, 8, 16, 31],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def pct_text(share: float) -> str:
    return f"{share * 100:.1f}%"


def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "start", fill: str = "#222", weight: str = "400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Arial, sans-serif" font-weight="{weight}">{html.escape(text)}</text>'
    )


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


class Canvas:
    def __init__(self, width: int, height: int, bg_hex: str) -> None:
        self.width = width
        self.height = height
        bg = hex_to_rgb(bg_hex)
        self.pixels = [[list(bg) for _ in range(width)] for _ in range(height)]

    def fill_rect(self, x: int, y: int, w: int, h: int, color_hex: str) -> None:
        color = hex_to_rgb(color_hex)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        for yy in range(y0, y1):
            row = self.pixels[yy]
            for xx in range(x0, x1):
                row[xx] = list(color)

    def draw_text(self, x: int, y: int, text: str, color_hex: str, scale: int = 2, anchor: str = "start") -> None:
        color = hex_to_rgb(color_hex)
        text_w = self.measure_text(text, scale)
        if anchor == "end":
            x -= text_w
        elif anchor == "middle":
            x -= text_w // 2
        cursor = x
        for char in text:
            glyph = FONT.get(char, FONT.get(char.lower(), FONT[" "]))
            for gy, bits in enumerate(glyph):
                for gx in range(5):
                    if bits & (1 << (4 - gx)):
                        px = cursor + gx * scale
                        py = y + gy * scale
                        for sy in range(scale):
                            yy = py + sy
                            if 0 <= yy < self.height:
                                row = self.pixels[yy]
                                for sx in range(scale):
                                    xx = px + sx
                                    if 0 <= xx < self.width:
                                        row[xx] = list(color)
            cursor += 6 * scale

    @staticmethod
    def measure_text(text: str, scale: int) -> int:
        return max(0, len(text) * 6 * scale - scale)

    def write_png(self, path: Path) -> None:
        raw = bytearray()
        for row in self.pixels:
            raw.append(0)
            for r, g, b in row:
                raw.extend((r, g, b))
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(chunk(b"IDAT", compressed))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def render_top_intents_svg(rows: list[dict[str, str]], out_path: Path, top_n: int) -> None:
    rows = sorted(rows, key=lambda r: int(r["count"]), reverse=True)[:top_n]
    width = 1080
    left = 270
    right = 120
    top = 70
    row_h = 42
    bar_h = 24
    chart_w = width - left - right
    height = top + len(rows) * row_h + 50

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        svg_text(40, 34, "Top intents", size=28, weight="700"),
        svg_text(40, 56, "Horizontal bar chart with percentage labels", size=14, fill=SUBTEXT),
    ]

    max_share = max(float(r["share"]) for r in rows) if rows else 1.0
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label_y = y + 18
        bar_y = y + 6
        share = float(row["share"])
        bar_w = 0 if max_share == 0 else chart_w * share / max_share
        parts.append(svg_text(left - 16, label_y, row["intent"], size=14, anchor="end"))
        parts.append(f'<rect x="{left}" y="{bar_y}" width="{chart_w}" height="{bar_h}" rx="4" fill="{TRACK}"/>')
        parts.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{TOP_BAR_COLOR}"/>')
        parts.append(svg_text(left + bar_w + 10, label_y, pct_text(share), size=13, fill="#333", weight="700"))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def render_top_intents_png(rows: list[dict[str, str]], out_path: Path, top_n: int) -> None:
    rows = sorted(rows, key=lambda r: int(r["count"]), reverse=True)[:top_n]
    width = 2160
    left = 540
    right = 220
    top = 140
    row_h = 84
    bar_h = 48
    chart_w = width - left - right
    height = top + len(rows) * row_h + 100
    canvas = Canvas(width, height, BG)
    canvas.draw_text(80, 28, "Top intents", TEXT, scale=6)
    canvas.draw_text(80, 84, "Horizontal bar chart with percentage labels", SUBTEXT, scale=4)
    max_share = max(float(r["share"]) for r in rows) if rows else 1.0
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        share = float(row["share"])
        bar_w = int(0 if max_share == 0 else chart_w * share / max_share)
        canvas.draw_text(left - 32, y + 16, row["intent"], TEXT, scale=4, anchor="end")
        canvas.fill_rect(left, y + 6, chart_w, bar_h, TRACK)
        canvas.fill_rect(left, y + 6, bar_w, bar_h, TOP_BAR_COLOR)
        canvas.draw_text(left + bar_w + 20, y + 16, pct_text(share), "#333333", scale=4)
    canvas.write_png(out_path)


def render_high_level_svg(rows: list[dict[str, str]], out_path: Path) -> None:
    by_cat = {row["category"]: float(row["share"]) for row in rows}
    width = 1200
    left = 60
    top = 92
    bar_h = 52
    chart_w = width - left - 60
    height = 250

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        svg_text(40, 36, "High-level category frequencies", size=28, weight="700"),
        svg_text(40, 60, "Horizontal stacked bar with fixed category order and percentage labels", size=14, fill=SUBTEXT),
    ]

    x = left
    y = top
    last_label_right = -10**9
    parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="6" fill="{TRACK}"/>')
    legend_y = y + bar_h + 44
    legend_x = left
    for cat in CATEGORY_ORDER:
        share = by_cat.get(cat, 0.0)
        seg_w = chart_w * share
        if seg_w > 0:
            parts.append(f'<rect x="{x}" y="{y}" width="{seg_w}" height="{bar_h}" fill="{CATEGORY_COLORS[cat]}"/>')
            label = pct_text(share)
            if seg_w >= 110:
                parts.append(svg_text(x + seg_w / 2, y + 31, label, size=14, anchor="middle", fill=WHITE, weight="700"))
            elif seg_w >= 60:
                parts.append(svg_text(x + seg_w / 2, y - 8, label, size=12, anchor="middle", fill="#333", weight="700"))
            est_w = len(cat) * 7
            label_center = x + seg_w / 2
            label_left = label_center - est_w / 2
            label_right = label_center + est_w / 2
            if label_left > last_label_right + 8:
                parts.append(svg_text(label_center, y + bar_h + 18, cat, size=12, anchor="middle", fill="#333"))
                last_label_right = label_right
        x += seg_w
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" rx="2" fill="{CATEGORY_COLORS[cat]}"/>')
        parts.append(svg_text(legend_x + 20, legend_y + 12, cat, size=13, fill=LABEL))
        legend_x += 120

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def render_high_level_png(rows: list[dict[str, str]], out_path: Path) -> None:
    by_cat = {row["category"]: float(row["share"]) for row in rows}
    width = 2400
    left = 120
    top = 184
    bar_h = 104
    chart_w = width - left - 120
    height = 500
    canvas = Canvas(width, height, BG)
    canvas.draw_text(80, 28, "High-level category frequencies", TEXT, scale=6)
    canvas.draw_text(80, 84, "Horizontal stacked bar with fixed category order and percentage labels", SUBTEXT, scale=4)
    canvas.fill_rect(left, top, chart_w, bar_h, TRACK)
    x = left
    last_label_right = -10**9
    legend_y = top + bar_h + 44
    legend_x = left
    for cat in CATEGORY_ORDER:
        share = by_cat.get(cat, 0.0)
        seg_w = int(chart_w * share)
        if seg_w > 0:
            canvas.fill_rect(x, top, seg_w, bar_h, CATEGORY_COLORS[cat])
            label = pct_text(share)
            if seg_w >= 220:
                canvas.draw_text(x + seg_w // 2, top + 36, label, WHITE, scale=4, anchor="middle")
            elif seg_w >= 120:
                canvas.draw_text(x + seg_w // 2, top - 32, label, "#333333", scale=4, anchor="middle")
            est_w = Canvas.measure_text(cat, 4)
            label_center = x + seg_w // 2
            label_left = label_center - est_w // 2
            label_right = label_center + est_w // 2
            if label_left > last_label_right + 16:
                canvas.draw_text(label_center, top + bar_h + 16, cat, "#333333", scale=4, anchor="middle")
                last_label_right = label_right
        canvas.fill_rect(legend_x, legend_y, 28, 28, CATEGORY_COLORS[cat])
        canvas.draw_text(legend_x + 40, legend_y, cat, LABEL, scale=4)
        legend_x += 240
        x += seg_w
    canvas.write_png(out_path)


def render_combined_high_level(
    model_rows: list[tuple[str, list[dict[str, str]]]], output_dir: Path
) -> tuple[Path, Path]:
    """Render one shared 100% stacked bar per model as SVG and PNG."""
    width = 1800
    left = 440
    right = 80
    top = 170
    row_h = 105
    bar_h = 62
    chart_w = width - left - right
    height = top + len(model_rows) * row_h + 165
    by_model = [{row["category"]: float(row["share"]) for row in rows} for _, rows in model_rows]
    step_counts = [sum(int(row["count"]) for row in rows) for _, rows in model_rows]
    svg_path = output_dir / "combined_high_level_stacked.svg"
    png_path = output_dir / "combined_high_level_stacked.png"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        svg_text(60, 54, "High-level category frequencies", size=32, weight="700"),
        svg_text(60, 88, "Horizontal stacked bars with shared category order and percentage labels", size=17, fill=SUBTEXT),
    ]
    canvas = Canvas(width, height, BG)
    canvas.draw_text(60, 28, "High-level category frequencies", TEXT, scale=6)
    canvas.draw_text(60, 84, "Horizontal stacked bars with shared category order and percentage labels", SUBTEXT, scale=3)

    for index, ((label, _), shares, steps) in enumerate(zip(model_rows, by_model, step_counts)):
        y = top + index * row_h
        x = left
        canvas_x = left
        canvas_y = y
        svg_label = f"{label} (n={steps})"
        svg_label_y = y + bar_h / 2 + 6
        svg_text_parts = svg_text(left - 28, svg_label_y, svg_label, size=18, anchor="end", fill=LABEL)
        parts.append(svg_text_parts)
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="6" fill="{TRACK}"/>')
        canvas.draw_text(canvas_x - 28, canvas_y + 24, svg_label, LABEL, scale=2, anchor="end")
        canvas.fill_rect(canvas_x, canvas_y, chart_w, bar_h, TRACK)
        for category in CATEGORY_ORDER:
            share = shares.get(category, 0.0)
            segment_w = chart_w * share
            canvas_w = int(segment_w)
            if segment_w:
                parts.append(f'<rect x="{x}" y="{y}" width="{segment_w}" height="{bar_h}" fill="{CATEGORY_COLORS[category]}"/>')
                canvas.fill_rect(canvas_x, canvas_y, canvas_w, bar_h, CATEGORY_COLORS[category])
                if segment_w >= 95:
                    label_text = pct_text(share)
                    parts.append(svg_text(x + segment_w / 2, y + 38, label_text, size=16, anchor="middle", fill=WHITE, weight="700"))
                    canvas.draw_text(canvas_x + canvas_w // 2, canvas_y + 22, label_text, WHITE, scale=3, anchor="middle")
            x += segment_w
            canvas_x += canvas_w

    legend_y = top + len(model_rows) * row_h + 30
    for index, category in enumerate(CATEGORY_ORDER):
        x = left + (index % 4) * 310
        y = legend_y + (index // 4) * 40
        parts.append(f'<rect x="{x}" y="{y}" width="18" height="18" rx="2" fill="{CATEGORY_COLORS[category]}"/>')
        parts.append(svg_text(x + 28, y + 15, category, size=16, fill=LABEL))
        canvas.fill_rect(x, y, 18, 18, CATEGORY_COLORS[category])
        canvas.draw_text(x + 28, y, category, LABEL, scale=3)

    parts.append("</svg>")
    svg_path.write_text("\n".join(parts))
    canvas.write_png(png_path)
    return svg_path, png_path


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text())
    output_dir = args.summary_json.parent
    intent_rows = read_csv(Path(summary["outputs"]["intent_counts_csv"]))
    high_rows = read_csv(Path(summary["outputs"]["high_level_counts_csv"]))

    top_svg = output_dir / "top_intents.svg"
    top_png = output_dir / "top_intents.png"
    high_svg = output_dir / "high_level_stacked.svg"
    high_png = output_dir / "high_level_stacked.png"

    render_top_intents_svg(intent_rows, top_svg, args.top_n)
    render_top_intents_png(intent_rows, top_png, args.top_n)
    render_high_level_svg(high_rows, high_svg)
    render_high_level_png(high_rows, high_png)

    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "top_intents_svg": str(top_svg),
                "top_intents_png": str(top_png),
                "high_level_stacked_svg": str(high_svg),
                "high_level_stacked_png": str(high_png),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

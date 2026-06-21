import sys
import os
import io
import csv
import json
import ast
import keyword
import re
import tempfile
import shutil
import ctypes
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib
from matplotlib import font_manager

def choose_default_font_name() -> str:
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
        if "Arial" in available:
            return "Arial"
    except Exception:
        pass
    return "DejaVu Sans"


DEFAULT_FONT_NAME = choose_default_font_name()
matplotlib.rcParams["font.family"] = [DEFAULT_FONT_NAME, "sans-serif"]
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import (
    is_color_like,
    ListedColormap,
    BoundaryNorm,
    LinearSegmentedColormap,
    to_rgba,
    to_rgb,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QWizard,
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QWidget,
    QTabWidget,
    QSplitter,
    QScrollArea,
    QFrame,
    QSizePolicy,
)


APP_NAME = "Elementti"
APP_VERSION = "1.0.8"
DEFAULT_SAMPLE_SEED = 42

APP_DEVELOPER_TEXT = (
    "Elementti was developed by the Laboratory of High Temperature Processes and Materials "
    "at Åbo Akademi University, Finland."
)

APP_CITATION_TEXT = (
    "If you use Elementti in your work, please cite the associated article."
)


def set_compact_layout(layout: QVBoxLayout) -> None:
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)


def add_compact_help_row(
    layout: QVBoxLayout,
    parent: QWidget,
    summary_text: str,
    help_title: str,
    help_text: str,
) -> None:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)

    label = QLabel(summary_text)
    label.setWordWrap(True)
    label.setStyleSheet("font-weight: 600;")

    help_button = QPushButton("Read instructions")
    help_button.setFixedWidth(130)
    help_button.setToolTip("Read detailed instructions for this step")
    help_button.clicked.connect(
        lambda _checked=False, p=parent, title=help_title, msg=help_text: QMessageBox.information(p, title, msg)
    )

    row.addWidget(label, 1)
    row.addWidget(help_button, 0)
    layout.addLayout(row)


def apply_inside_top_legend_with_headroom(
    ax,
    max_columns: int = 4,
    legend_handles: Optional[List[object]] = None,
    legend_labels: Optional[List[str]] = None,
    place_legend: bool = True,
    extra_axes: Optional[List[object]] = None,
    skip_headroom_axes: Optional[List[object]] = None,
) -> None:
    """Place a compact inside legend and reserve visual headroom above line-profile data.

    For dual-y-axis line profiles, pass the secondary axis in extra_axes. The
    legend combines curves from all axes, while headroom is applied separately
    to each axis unless it has manual y-limits.
    """
    axes = [ax] + [axis for axis in (extra_axes or []) if axis is not None]

    if legend_handles is None or legend_labels is None:
        handles = []
        labels = []
        for axis in axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            handles.extend(axis_handles)
            labels.extend(axis_labels)
    else:
        handles, labels = list(legend_handles), list(legend_labels)

    clean_pairs = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not str(label).startswith("_")
    ]

    n_legend_items = len(clean_pairs)
    ncol = max(1, min(n_legend_items if n_legend_items else 1, max_columns))
    legend_rows = int(np.ceil(n_legend_items / ncol)) if n_legend_items else 1
    skip_ids = {id(axis) for axis in (skip_headroom_axes or []) if axis is not None}

    def apply_headroom(target_ax) -> None:
        if id(target_ax) in skip_ids:
            return

        finite_values = []
        for line in target_ax.lines:
            try:
                y_values = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            finite = y_values[np.isfinite(y_values)]
            if finite.size:
                finite_values.append(finite)

        if not finite_values:
            return

        all_values = np.concatenate(finite_values)
        data_min = float(np.min(all_values))
        data_max = float(np.max(all_values))
        old_bottom, _old_top = target_ax.get_ylim()
        data_span = data_max - data_min
        if not np.isfinite(data_span) or data_span <= 0:
            data_span = max(abs(data_max), 1.0)

        # Use only enough extra headroom to fit the legend inside the axes.
        # Tick labels are not removed above the data maximum; this keeps the
        # full visible y-axis scale readable.
        headroom_fraction = min(0.20, 0.105 + 0.045 * max(0, legend_rows - 1))
        headroom = max(data_span * headroom_fraction, abs(data_max) * 0.012, 1e-9)
        new_top = data_max + headroom

        new_bottom = old_bottom if np.isfinite(old_bottom) and old_bottom < new_top else data_min - 0.05 * data_span
        target_ax.set_ylim(new_bottom, new_top)

    for axis in axes:
        apply_headroom(axis)

    if not place_legend or not clean_pairs:
        return

    handles, labels = zip(*clean_pairs)
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderaxespad=0.0,
        fontsize="small",
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass
    try:
        legend.set_zorder(10000)
    except Exception:
        pass


def add_inside_top_legend_only(ax, handles, labels, max_columns: int = 4) -> None:
    clean_pairs = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not str(label).startswith("_")
    ]
    if not clean_pairs:
        return

    handles, labels = zip(*clean_pairs)
    ncol = max(1, min(len(handles), max_columns))
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderaxespad=0.0,
        fontsize="small",
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass
    try:
        legend.set_zorder(10000)
    except Exception:
        pass


def normalize_line_profile_legend_position(position: str) -> str:
    value = str(position).strip().lower()
    if value in {"outside right", "right outside", "outside", "right"}:
        return "Outside right"
    return "Inside plot"


def add_outside_right_legend(ax, handles, labels, right_axis_present: bool = False) -> None:
    clean_pairs = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not str(label).startswith("_")
    ]
    if not clean_pairs:
        return

    handles, labels = zip(*clean_pairs)
    # A right y-axis needs extra room for its tick labels and axis title.
    anchor_x = 1.22 if right_axis_present else 1.03
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(anchor_x, 1.0),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderaxespad=0.0,
        fontsize="small",
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.45,
    )
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass
    try:
        legend.set_zorder(10000)
    except Exception:
        pass


def format_colorbar_tick_value(value: float) -> str:
    return f"{float(value):.9g}"


def floats_close(a: float, b: float) -> bool:
    scale = max(abs(float(a)), abs(float(b)), 1.0)
    return abs(float(a) - float(b)) <= scale * 1e-9


def dedupe_sorted_floats(values: List[float]) -> List[float]:
    clean = sorted(float(v) for v in values if np.isfinite(v))
    result: List[float] = []
    for value in clean:
        if not result or not floats_close(value, result[-1]):
            result.append(value)
    return result


def set_colorbar_ticks_with_clipping_labels(
    cbar,
    vmin: float,
    vmax: float,
    actual_min: float,
    actual_max: float,
    force_min_tick: bool = False,
    force_max_tick: bool = False,
) -> None:
    """Show endpoint ticks and mark clipped continuous colorbar values with ≤ / ≥."""
    if not all(np.isfinite(v) for v in [vmin, vmax, actual_min, actual_max]):
        return
    if vmin >= vmax:
        return

    current_ticks = [float(tick) for tick in cbar.get_ticks() if np.isfinite(tick)]
    ticks = [tick for tick in current_ticks if vmin <= tick <= vmax]

    clips_low = actual_min < vmin and not floats_close(actual_min, vmin)
    clips_high = actual_max > vmax and not floats_close(actual_max, vmax)

    if force_min_tick or clips_low:
        ticks.append(vmin)
    if force_max_tick or clips_high:
        ticks.append(vmax)

    ticks = dedupe_sorted_floats(ticks)
    if not ticks:
        ticks = [vmin, vmax]

    cbar.set_ticks(ticks)
    labels = []
    for tick in ticks:
        label = format_colorbar_tick_value(tick)
        if floats_close(tick, vmin) and clips_low:
            label = "≤" + label
        if floats_close(tick, vmax) and clips_high:
            label = "≥" + label
        labels.append(label)
    cbar.ax.set_yticklabels(labels)


def set_manual_bin_colorbar_ticks_with_clipping_labels(
    cbar,
    boundaries: List[float],
    actual_min: float,
    actual_max: float,
) -> None:
    """Mark clipped manual-bin colorbar boundary labels with ≤ / ≥."""
    if len(boundaries) < 2:
        return
    finite_boundaries = [float(value) for value in boundaries if np.isfinite(value)]
    if len(finite_boundaries) < 2:
        return

    low = finite_boundaries[0]
    high = finite_boundaries[-1]
    clips_low = actual_min < low and not floats_close(actual_min, low)
    clips_high = actual_max > high and not floats_close(actual_max, high)

    cbar.set_ticks(finite_boundaries)
    labels = []
    for idx, value in enumerate(finite_boundaries):
        label = format_colorbar_tick_value(value)
        if idx == 0 and clips_low:
            label = "≤" + label
        if idx == len(finite_boundaries) - 1 and clips_high:
            label = "≥" + label
        labels.append(label)
    cbar.ax.set_yticklabels(labels)


def set_page_layout(page: QWizardPage, content_layout: QVBoxLayout, scrollable: bool = True) -> None:
    if not scrollable:
        page.setLayout(content_layout)
        return

    container = QWidget()
    container.setLayout(content_layout)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    scroll_area.setWidget(container)

    outer = QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(scroll_area)
    page.setLayout(outer)


def resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


def suggest_name(file_name: str, used_names: set) -> str:
    base = os.path.splitext(file_name)[0].strip()
    lower = base.lower()

    hints = {
        "chlorine": "Cl",
        "iron": "Fe",
        "oxygen": "O",
        "carbon": "C",
        "calcium": "Ca",
        "phosphorus": "P",
        "potassium": "K",
        "sodium": "Na",
        "magnesium": "Mg",
        "sulfur": "S",
        "sulphur": "S",
        "silicon": "Si",
        "aluminum": "Al",
        "aluminium": "Al",
        "manganese": "Mn",
        "nickel": "Ni",
        "chromium": "Cr",
        "zinc": "Zn",
        "copper": "Cu",
        "gray": "Grey",
        "grayscale": "Grey",
        "greyscale": "Grey",
    }

    suggested = None
    for key, short_name in hints.items():
        if key in lower:
            suggested = short_name
            break

    if suggested is None:
        suggested = base.replace(" ", "_")

    if suggested not in used_names:
        return suggested

    counter = 2
    while f"{suggested}_{counter}" in used_names:
        counter += 1
    return f"{suggested}_{counter}"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name).strip().strip(".")
    return cleaned if cleaned else "output"


def make_unique_output_path(
    folder: str,
    file_stub: str,
    extension: str,
    reserved_paths: set,
) -> str:
    """Return a safe, non-overwriting output path.

    Output names can contain scientific notation such as A/B, A:B, or
    formulas. Those names are sanitized for the file system, which can make
    different outputs collapse to the same filename. This helper reserves each
    generated path and also checks existing files, so exports never silently
    overwrite another result from the same run or a previous run in the same
    folder.
    """
    base_stub = sanitize_filename(file_stub)
    ext = str(extension).strip().lstrip(".")
    if not ext:
        raise ValueError("Output file extension is empty.")

    counter = 1
    while True:
        candidate_stub = base_stub if counter == 1 else f"{base_stub}_{counter}"
        candidate_path = os.path.join(folder, f"{candidate_stub}.{ext}")
        normalized = os.path.normcase(os.path.abspath(candidate_path))
        if normalized not in reserved_paths and not os.path.exists(candidate_path):
            reserved_paths.add(normalized)
            return candidate_path
        counter += 1


def is_safe_demo_sample_folder(folder: str) -> bool:
    folder_text = str(folder).strip()
    if not folder_text:
        return False

    try:
        folder_abs = os.path.abspath(folder_text)
        temp_abs = os.path.abspath(tempfile.gettempdir())
        return (
            os.path.isdir(folder_abs)
            and os.path.basename(folder_abs).startswith("elementti_demo_")
            and os.path.commonpath([folder_abs, temp_abs]) == temp_abs
        )
    except Exception:
        return False


def cleanup_demo_sample_folder(folder: str) -> None:
    if not is_safe_demo_sample_folder(folder):
        return
    try:
        shutil.rmtree(os.path.abspath(folder), ignore_errors=True)
    except Exception:
        pass


def path_is_inside_folder(path: str, folder: str) -> bool:
    try:
        path_abs = os.path.abspath(path)
        folder_abs = os.path.abspath(folder)
        return os.path.commonpath([path_abs, folder_abs]) == folder_abs
    except Exception:
        return False


def cleanup_unused_demo_sample_folder(state) -> None:
    folder = getattr(state, "sample_data_folder", "")
    if not folder:
        return

    still_used = any(
        path_is_inside_folder(path, folder)
        for path in getattr(state, "selected_file_paths", [])
    )
    if still_used:
        return

    cleanup_demo_sample_folder(folder)
    state.sample_data_folder = ""
    state.sample_data_seed = None
    if getattr(state, "sample_data_mode", "") == "demo":
        state.sample_data_mode = ""


def find_app_icon_path() -> str:
    base_dir = os.path.dirname(resource_path("__elementti_resource_probe__"))
    preferred_names = [
        "icoElementti.ico",
        f"icoElementti_{APP_VERSION.replace('.', '_')}.ico",
    ]

    for name in preferred_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path

    try:
        for name in sorted(os.listdir(base_dir)):
            lower = name.lower()
            if lower.startswith("icoelementti") and lower.endswith(".ico"):
                path = os.path.join(base_dir, name)
                if os.path.exists(path):
                    return path
    except Exception:
        pass

    return ""


def parse_float_user_text(value_text: str, field_name: str = "Value") -> float:
    """Parse a user-entered number from common international formats.

    Supported examples include 1.874568, 1,874568, 1,234.56,
    1.234,56, 1 234,56, 1'234.56, and full-width digit forms.
    """
    text = unicodedata.normalize("NFKC", str(value_text))
    text = text.strip()
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[\s']+", "", text)

    if text == "":
        raise ValueError(f"{field_name} is empty.")

    # Accept either comma or dot as decimal separator. If both appear, the
    # rightmost separator is interpreted as the decimal separator and the
    # other separator is treated as a thousands/grouping separator.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) > 2:
            text = "".join(parts[:-1]) + "." + parts[-1]
        else:
            text = text.replace(",", ".")
    elif text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid number.")


def split_numeric_list_text(list_text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", str(list_text)).strip()
    if text == "":
        return []
    # Semicolon is recommended when decimal commas are used, e.g. 0,5; 1,0.
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    return [p.strip() for p in text.split(",") if p.strip()]


def parse_bins_text(bins_text: str) -> List[float]:
    parts = split_numeric_list_text(bins_text)
    return [parse_float_user_text(p, "Manual-bin boundary") for p in parts]


def parse_color_list(color_text: str) -> List[str]:
    return [c.strip() for c in color_text.split(",") if c.strip()]


def format_detected_float(value: float) -> str:
    return f"{float(value):.9g}"


def format_plot_number(value: float) -> str:
    """Format axis/colorbar numbers compactly while preserving meaningful precision."""
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.6g}"


def decimal_places_for_tick_spacing(spacing: float, max_decimals: int = 6) -> int:
    """Return a sensible decimal count that matches nearby automatic ticks."""
    try:
        spacing = abs(float(spacing))
    except Exception:
        return 3
    if not np.isfinite(spacing) or spacing <= 0:
        return 3
    for decimals in range(max_decimals + 1):
        rounded = round(spacing, decimals)
        if abs(rounded - spacing) <= max(abs(spacing), 1.0) * 1e-8:
            return decimals
    return max_decimals


def format_endpoint_number_like_ticks(value: float, typical_spacing: float, endpoint: str) -> str:
    """Round automatic colorbar endpoints outward so labels match nearby ticks.

    For example, if the automatic ticks are spaced by 0.2 and the processed
    maximum is 1.78378, the displayed endpoint label becomes 1.8 instead of
    1.78378. Manual user-entered endpoints are kept exact elsewhere.
    """
    value = float(value)
    if not np.isfinite(value):
        return format_plot_number(value)

    decimals = decimal_places_for_tick_spacing(typical_spacing)
    factor = 10.0 ** decimals
    tolerance = max(abs(value), 1.0) * 1e-10

    if endpoint == "upper":
        rounded = np.ceil((value - tolerance) * factor) / factor
    elif endpoint == "lower":
        rounded = np.floor((value + tolerance) * factor) / factor
    else:
        rounded = round(value, decimals)

    if factor > 0 and abs(rounded) < 0.5 / factor:
        rounded = 0.0
    return format_plot_number(rounded)


def build_endpoint_aware_tick_labels(
    current_ticks: List[float],
    lower_value: float,
    upper_value: float,
    include_lower: bool = True,
    include_upper: bool = True,
    lower_label: Optional[str] = None,
    upper_label: Optional[str] = None,
    round_lower_to_tick_spacing: bool = False,
    round_upper_to_tick_spacing: bool = False,
) -> Tuple[List[float], List[str]]:
    """Add endpoint ticks, but remove automatic ticks that are visually too close."""
    try:
        lower_value = float(lower_value)
        upper_value = float(upper_value)
    except Exception:
        return [], []

    if not np.isfinite(lower_value) or not np.isfinite(upper_value) or lower_value >= upper_value:
        return [], []

    span = upper_value - lower_value
    tolerance = max(abs(span), 1.0) * 1e-9

    in_range_ticks = [
        float(tick)
        for tick in current_ticks
        if np.isfinite(tick) and lower_value <= float(tick) <= upper_value
    ]
    automatic_ticks = dedupe_sorted_floats(in_range_ticks)

    tick_pool = list(automatic_ticks)
    if include_lower:
        tick_pool.append(lower_value)
    if include_upper:
        tick_pool.append(upper_value)
    tick_pool = dedupe_sorted_floats(tick_pool)

    automatic_diffs = [
        automatic_ticks[i + 1] - automatic_ticks[i]
        for i in range(len(automatic_ticks) - 1)
        if automatic_ticks[i + 1] - automatic_ticks[i] > tolerance
    ]
    if automatic_diffs:
        typical_spacing = float(np.median(automatic_diffs))
    else:
        typical_spacing = span / max(4.0, float(max(len(tick_pool) - 1, 1)))

    # A little over one third of a normal tick step removes redundant labels
    # near a non-round endpoint, e.g. it keeps ≥3.1 and drops the nearby 3.
    endpoint_clearance = max(typical_spacing * 0.36, span * 0.012, tolerance * 10.0)

    cleaned_ticks: List[float] = []
    for tick in tick_pool:
        is_lower_endpoint = include_lower and abs(tick - lower_value) <= tolerance
        is_upper_endpoint = include_upper and abs(tick - upper_value) <= tolerance
        if not is_lower_endpoint and not is_upper_endpoint:
            if include_lower and tick - lower_value < endpoint_clearance:
                continue
            if include_upper and upper_value - tick < endpoint_clearance:
                continue
        cleaned_ticks.append(tick)

    if include_lower and not any(abs(tick - lower_value) <= tolerance for tick in cleaned_ticks):
        cleaned_ticks.append(lower_value)
    if include_upper and not any(abs(tick - upper_value) <= tolerance for tick in cleaned_ticks):
        cleaned_ticks.append(upper_value)
    cleaned_ticks = dedupe_sorted_floats(cleaned_ticks)

    labels: List[str] = []
    for tick in cleaned_ticks:
        if include_lower and abs(tick - lower_value) <= tolerance:
            if lower_label is not None:
                labels.append(lower_label)
            elif round_lower_to_tick_spacing:
                labels.append(format_endpoint_number_like_ticks(lower_value, typical_spacing, "lower"))
            else:
                labels.append(format_plot_number(lower_value))
        elif include_upper and abs(tick - upper_value) <= tolerance:
            if upper_label is not None:
                labels.append(upper_label)
            elif round_upper_to_tick_spacing:
                labels.append(format_endpoint_number_like_ticks(upper_value, typical_spacing, "upper"))
            else:
                labels.append(format_plot_number(upper_value))
        else:
            labels.append(format_plot_number(tick))

    return cleaned_ticks, labels


def set_colorbar_clipping_endpoint_labels(
    cbar,
    lower_value: float,
    upper_value: float,
    show_lower_as_clipped: bool = False,
    show_upper_as_clipped: bool = False,
    round_unclipped_lower: bool = False,
    round_unclipped_upper: bool = False,
) -> None:
    """Ensure colorbar end ticks are present without clutter near the endpoints."""
    lower_label = f"≤{format_plot_number(lower_value)}" if show_lower_as_clipped else None
    upper_label = f"≥{format_plot_number(upper_value)}" if show_upper_as_clipped else None

    try:
        current_ticks = list(cbar.get_ticks())
    except Exception:
        current_ticks = []

    ticks, labels = build_endpoint_aware_tick_labels(
        current_ticks,
        lower_value,
        upper_value,
        include_lower=True,
        include_upper=True,
        lower_label=lower_label,
        upper_label=upper_label,
        round_lower_to_tick_spacing=round_unclipped_lower and not show_lower_as_clipped,
        round_upper_to_tick_spacing=round_unclipped_upper and not show_upper_as_clipped,
    )
    if not ticks:
        return

    try:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(axis="y", pad=2)
    except Exception:
        pass


def set_y_axis_endpoint_labels_for_manual_limits(ax, min_text: str, max_text: str) -> None:
    """Show manual y-axis endpoints and mark clipped curves with ≤ / ≥."""
    include_lower = str(min_text).strip() != ""
    include_upper = str(max_text).strip() != ""
    if not include_lower and not include_upper:
        return

    lower_value, upper_value = ax.get_ylim()
    if lower_value >= upper_value:
        lower_value, upper_value = upper_value, lower_value

    finite_values = []
    for line in ax.lines:
        try:
            values = np.asarray(line.get_ydata(), dtype=float)
        except Exception:
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)

    if finite_values:
        all_values = np.concatenate(finite_values)
        actual_min = float(np.min(all_values))
        actual_max = float(np.max(all_values))
    else:
        actual_min = lower_value
        actual_max = upper_value

    lower_clipped = actual_min < lower_value and not floats_close(actual_min, lower_value)
    upper_clipped = actual_max > upper_value and not floats_close(actual_max, upper_value)
    lower_label = f"≤{format_plot_number(lower_value)}" if lower_clipped else format_plot_number(lower_value)
    upper_label = f"≥{format_plot_number(upper_value)}" if upper_clipped else format_plot_number(upper_value)

    ticks, labels = build_endpoint_aware_tick_labels(
        list(ax.get_yticks()),
        lower_value,
        upper_value,
        include_lower=include_lower,
        include_upper=include_upper,
        lower_label=lower_label,
        upper_label=upper_label,
    )
    if not ticks:
        return

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", pad=2)


# Pixel sizes are stored internally as micrometres per pixel (µm/pixel),
# but metadata and users may write them in other length units. Keep the
# accepted units intentionally limited to common microscopy/image-calibration
# units so Elementti does not silently interpret unrelated metadata as a pixel
# size.
PIXEL_SIZE_NUMBER_PATTERN = (
    r"(?P<value>[+-]?(?:\d+(?:[ \t.,'’]\d+)*|[.,]\d+)(?:[eE][+-]?\d+)?)"
)
PIXEL_SIZE_UNIT_PATTERN = (
    r"(?<![A-Za-zµμ])(?P<unit>"
    r"nanometers?|nanometres?|nm|"
    r"micrometers?|micrometres?|microns?|µm|μm|um|"
    r"millimeters?|millimetres?|mm|"
    r"centimeters?|centimetres?|cm|"
    r"meters?|metres?|m"
    r")"
    r"(?![A-Za-zµμ])"
)
PIXEL_SIZE_KEYWORD_PATTERN = (
    r"(?:pixel\s*(?:size|width|height|spacing|resolution)|"
    r"pix(?:el)?\s*size|step\s*size|calibration|resolution)"
)


def normalize_pixel_size_unit(unit_text: str) -> str:
    unit = unicodedata.normalize("NFKC", str(unit_text)).strip().lower().replace("μ", "µ")
    unit = unit.rstrip(".:")

    if unit in ("nm", "nanometer", "nanometers", "nanometre", "nanometres"):
        return "nm"
    if unit in ("µm", "um", "micrometer", "micrometers", "micrometre", "micrometres", "micron", "microns"):
        return "µm"
    if unit in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return "mm"
    if unit in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return "cm"
    if unit in ("m", "meter", "meters", "metre", "metres"):
        return "m"

    raise ValueError(f"Unsupported pixel-size unit '{unit_text}'.")


def pixel_size_unit_factor_to_um(unit_text: str) -> float:
    unit = normalize_pixel_size_unit(unit_text)
    factors = {
        "nm": 0.001,
        "µm": 1.0,
        "mm": 1000.0,
        "cm": 10000.0,
        "m": 1000000.0,
    }
    return factors[unit]


def parse_pixel_size_text_to_um(value_text: str, field_name: str = "Pixel size") -> float:
    """Parse a pixel size and return micrometres per pixel.

    Plain numbers are interpreted as µm/pixel for backward compatibility.
    Unit-aware entries such as 1874 nm/pixel, 1.874 µm, 0.001874 mm/px,
    or 1.874 um per pixel are converted to µm/pixel.
    """
    text = unicodedata.normalize("NFKC", str(value_text)).replace("μ", "µ").strip()
    if text == "":
        raise ValueError(f"{field_name} is empty.")

    try:
        return parse_float_user_text(text, field_name)
    except ValueError:
        pass

    unit_after_value = re.search(
        PIXEL_SIZE_NUMBER_PATTERN + r"\s*" + PIXEL_SIZE_UNIT_PATTERN + r"(?:\s*(?:/|per)\s*(?:pixel|px))?",
        text,
        re.I,
    )
    if unit_after_value:
        value = parse_float_user_text(unit_after_value.group("value"), field_name)
        return value * pixel_size_unit_factor_to_um(unit_after_value.group("unit"))

    unit_before_value = re.search(
        PIXEL_SIZE_UNIT_PATTERN + r"\s*(?:/\s*(?:pixel|px))?[^\n\r0-9+-]{0,40}" + PIXEL_SIZE_NUMBER_PATTERN,
        text,
        re.I,
    )
    if unit_before_value:
        value = parse_float_user_text(unit_before_value.group("value"), field_name)
        return value * pixel_size_unit_factor_to_um(unit_before_value.group("unit"))

    raise ValueError(
        f"{field_name} must be a positive number, optionally followed by nm, µm/um, mm, cm, or m."
    )


def detect_pixel_size_um_from_file(file_path: str) -> str:
    """Try to detect and convert pixel size from CSV metadata/header text.

    The returned value is always formatted as µm/pixel. Recognized metadata may
    use nm, µm/um, mm, cm, or m. Pure numeric map data without metadata will
    return an empty string.
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(65536)
    except Exception:
        return ""

    text = raw.replace(b"\xa0", b" ").decode("utf-8", errors="ignore")
    text = unicodedata.normalize("NFKC", text).replace("μ", "µ")

    number = PIXEL_SIZE_NUMBER_PATTERN
    unit = PIXEL_SIZE_UNIT_PATTERN
    keyword = PIXEL_SIZE_KEYWORD_PATTERN

    patterns = [
        # Common form: Pixel Size 1874.568 nm/pixel
        re.compile(keyword + r"[^\n\r0-9+-]{0,120}" + number + r"\s*" + unit + r"(?:\s*(?:/|per)\s*(?:pixel|px))?", re.I),
        # Common form without a keyword but with explicit per-pixel unit.
        re.compile(number + r"\s*" + unit + r"\s*(?:/|per)\s*(?:pixel|px)", re.I),
        # Some exports write the unit before the value, e.g. Pixel size [nm] 1874.568.
        re.compile(keyword + r"[^\n\r]{0,80}?" + unit + r"[^\n\r0-9+-]{0,80}" + number, re.I),
        # Value and unit before the keyword, e.g. 1874.568 nm pixel size.
        re.compile(number + r"\s*" + unit + r"[^\n\r]{0,80}" + keyword, re.I),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = parse_float_user_text(match.group("value"), "Detected pixel size")
            value_um = value * pixel_size_unit_factor_to_um(match.group("unit"))
        except ValueError:
            continue
        if value_um > 0:
            return format_detected_float(value_um)

    # Backward-compatible fallback: if metadata clearly says pixel size but does
    # not state a unit, assume the value is already in µm/pixel. Users still see
    # and can edit the value on the preprocessing page before export.
    legacy_pattern = re.compile(keyword + r"[^\n\r0-9+-]{0,120}" + number, re.I)
    match = legacy_pattern.search(text)
    if match:
        try:
            value = parse_float_user_text(match.group("value"), "Detected pixel size")
        except ValueError:
            value = 0.0
        if value > 0:
            return format_detected_float(value)

    return ""


GRAYSCALE_CONDITIONS = ["below", "above", "between", "outside"]
CONDITIONAL_OPERATORS = ["above", "below", "between"]


def normalize_grayscale_condition(condition: str) -> str:
    value = condition.strip().lower()
    if value in ("below", "<="):
        return "below"
    if value in ("above", ">="):
        return "above"
    if value == "between":
        return "between"
    if value == "outside":
        return "outside"
    return condition.strip()


def normalize_conditional_operator(operator: str) -> str:
    value = operator.strip().lower()
    if value in (">", ">=", "above"):
        return "above"
    if value in ("<", "<=", "below"):
        return "below"
    if value == "between":
        return "between"
    return operator.strip()


def is_grayscale_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("grey", "gray", "grayscale", "greyscale"))


def default_colorbar_label(output_name: str) -> str:
    return output_name


SINGLE_HUE_COLORMAP_NAMES = {
    "green",
    "red",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "brown",
}


def build_single_hue_colormap(color_name: str):
    r, g, b = to_rgb(color_name)
    dark = (r * 0.15, g * 0.15, b * 0.15)
    bright = (r, g, b)
    return LinearSegmentedColormap.from_list(
        f"single_hue_{color_name}",
        [dark, bright],
        N=256,
    )


def get_continuous_colormap(colormap_name: str):
    name = colormap_name.strip().lower()
    if name in SINGLE_HUE_COLORMAP_NAMES:
        return build_single_hue_colormap(name)
    return plt.get_cmap(colormap_name)


def parse_manual_index_list(text: str, what: str) -> List[int]:
    cleaned = str(text).strip()
    if cleaned == "":
        return []

    # Accept both compact ranges and individual indices, for example:
    # [1-5], 17, 28, [94-102], 140, 143
    # Brackets are optional, so 1-5 is also accepted.
    items = [x.strip() for x in re.split(r"[,;]+", cleaned) if x.strip()]
    result: List[int] = []

    for item in items:
        item = item.strip()
        item_no_brackets = item
        if item_no_brackets.startswith("[") and item_no_brackets.endswith("]"):
            item_no_brackets = item_no_brackets[1:-1].strip()

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", item_no_brackets)
        if range_match:
            start_value = int(range_match.group(1))
            end_value = int(range_match.group(2))
            if start_value <= 0 or end_value <= 0:
                raise ValueError(f"{what} ranges must use 1-based positive numbers, like [1-5].")
            if start_value > end_value:
                raise ValueError(f"{what} range '{item}' is invalid. The first number must be smaller than the second.")
            result.extend(value - 1 for value in range(start_value, end_value + 1))
            continue

        try:
            value = int(item_no_brackets)
        except ValueError:
            raise ValueError(
                f"{what} must use whole numbers or ranges like [1-5], 17, [94-102]."
            )

        if value <= 0:
            raise ValueError(f"{what} must use 1-based positive numbers, like 1, 2, or [1-5].")

        result.append(value - 1)

    return sorted(set(result))

# ---------------------------------------------------------------------------
# Custom formula outputs
# ---------------------------------------------------------------------------
# Formula maps are evaluated with a small safe parser instead of Python eval().
# Supported syntax: input-map names, [Exact input name] or {Exact input name}, numeric constants,
# parentheses, +, -, *, /, **, unary +/- and a small set of NumPy functions.

FORMULA_ALLOWED_FUNCTIONS = {
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "exp": np.exp,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
}

FORMULA_FUNCTION_ARITY = {
    "sqrt": 1,
    "log": 1,
    "ln": 1,
    "log10": 1,
    "log2": 1,
    "exp": 1,
    "abs": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "minimum": 2,
    "maximum": 2,
    "min": 2,
    "max": 2,
    "clip": 3,
}

FORMULA_ALLOWED_CONSTANTS = {
    "pi": float(np.pi),
    "e": float(np.e),
    "nan": float("nan"),
}


def format_formula_allowed_functions() -> str:
    return ", ".join(f"{name}()" for name in sorted(FORMULA_ALLOWED_FUNCTIONS))

FORMULA_BARE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FORMULA_INTERNAL_VARIABLE_PREFIX = "__elementti_formula_var_"
FORMULA_MAX_EXPRESSION_LENGTH = 500


@dataclass
class CompiledFormula:
    original_expression: str
    prepared_expression: str
    tree: ast.Expression
    token_to_name: Dict[str, str]
    referenced_names: List[str]


def is_formula_bare_name_allowed(name: str) -> bool:
    """Return True if an input map can safely be used without [brackets]."""
    if not FORMULA_BARE_NAME_RE.fullmatch(name):
        return False
    if keyword.iskeyword(name):
        return False
    if name in FORMULA_ALLOWED_FUNCTIONS or name in FORMULA_ALLOWED_CONSTANTS:
        return False
    return True


def build_formula_name_tokens(available_names: List[str]) -> Dict[str, str]:
    return {
        name: f"{FORMULA_INTERNAL_VARIABLE_PREFIX}{index}"
        for index, name in enumerate(available_names)
    }


def prepare_formula_expression(expression: str, available_names: List[str]) -> Tuple[str, Dict[str, str]]:
    text = unicodedata.normalize("NFKC", str(expression)).strip()
    if text == "":
        raise ValueError("Formula expression is empty.")
    if len(text) > FORMULA_MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Formula expression is too long. Keep it under {FORMULA_MAX_EXPRESSION_LENGTH} characters.")

    name_to_token = build_formula_name_tokens(available_names)

    def replace_named_reference(match: re.Match) -> str:
        # Users may write either [Exact map name] or {{Exact map name}}.
        # Square brackets are shown in the UI; braces are accepted as a readable alias.
        map_name = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        if map_name not in name_to_token:
            available = ", ".join(available_names) if available_names else "none"
            raise ValueError(
                f"Formula uses unknown map name '{map_name}'. Available input maps: {available}."
            )
        return name_to_token[map_name]

    prepared = re.sub(r"\[([^\[\]]+)\]|\{([^{}]+)\}", replace_named_reference, text)
    if "[" in prepared or "]" in prepared:
        raise ValueError("Formula has unmatched square brackets. Use [Exact map name] for names with spaces or symbols.")
    if "{" in prepared or "}" in prepared:
        raise ValueError("Formula has unmatched braces. Use {Exact map name} for names with spaces or symbols.")

    # Replace bare input names with internal tokens. Sort by length to avoid
    # replacing a shorter name inside a longer one, e.g. Fe inside Fe_total.
    for name in sorted(available_names, key=len, reverse=True):
        if not is_formula_bare_name_allowed(name):
            continue
        token = name_to_token[name]
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
        prepared = re.sub(pattern, token, prepared)

    token_to_name = {token: name for name, token in name_to_token.items()}
    return prepared, token_to_name


class FormulaAstValidator(ast.NodeVisitor):
    def __init__(self, token_to_name: Dict[str, str]):
        self.token_to_name = token_to_name
        self.referenced_tokens = set()

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            if isinstance(node.op, ast.BitXor):
                raise ValueError("Formula uses '^'. Use '**' for powers, for example X**2.")
            raise ValueError("Formula supports only +, -, *, /, and ** operators.")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Formula supports only unary + and unary -.")
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Formula constants must be numbers.")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.token_to_name:
            self.referenced_tokens.add(node.id)
            return
        if node.id in FORMULA_ALLOWED_CONSTANTS:
            return
        if node.id in FORMULA_ALLOWED_FUNCTIONS:
            raise ValueError(f"Formula function '{node.id}' must be called with parentheses.")
        raise ValueError(
            f"Formula uses unknown name '{node.id}'. Use input-map names, [Exact map name] or {{Exact map name}}, "
            f"numbers, or approved functions: {format_formula_allowed_functions()}."
        )

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in FORMULA_ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Formula functions are limited to the approved functions: {format_formula_allowed_functions()}."
            )
        if node.keywords:
            raise ValueError("Formula functions do not accept keyword arguments.")
        expected = FORMULA_FUNCTION_ARITY[node.func.id]
        if len(node.args) != expected:
            raise ValueError(f"Formula function '{node.func.id}' expects {expected} argument(s).")
        for arg in node.args:
            self.visit(arg)

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Formula contains unsupported syntax: {type(node).__name__}.")


def compile_formula_expression(expression: str, available_names: List[str]) -> CompiledFormula:
    prepared, token_to_name = prepare_formula_expression(expression, available_names)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Formula syntax is invalid: {exc.msg}.")

    validator = FormulaAstValidator(token_to_name)
    validator.visit(tree)

    name_to_token = build_formula_name_tokens(available_names)
    referenced_names = [
        name
        for name in available_names
        if name_to_token[name] in validator.referenced_tokens
    ]
    if not referenced_names:
        raise ValueError("Formula must reference at least one input map.")

    return CompiledFormula(
        original_expression=str(expression).strip(),
        prepared_expression=prepared,
        tree=tree,
        token_to_name=token_to_name,
        referenced_names=referenced_names,
    )


class FormulaEvaluator(ast.NodeVisitor):
    def __init__(self, compiled: CompiledFormula, arrays_by_name: Dict[str, np.ndarray]):
        self.compiled = compiled
        self.arrays_by_name = arrays_by_name

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError("Formula supports only +, -, *, /, and ** operators.")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Formula supports only unary + and unary -.")

    def visit_Constant(self, node: ast.Constant):
        return float(node.value)

    def visit_Name(self, node: ast.Name):
        if node.id in self.compiled.token_to_name:
            map_name = self.compiled.token_to_name[node.id]
            if map_name not in self.arrays_by_name:
                raise ValueError(f"Formula input map '{map_name}' was not loaded.")
            return self.arrays_by_name[map_name]
        if node.id in FORMULA_ALLOWED_CONSTANTS:
            return FORMULA_ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Formula uses unknown name '{node.id}'.")

    def visit_Call(self, node: ast.Call):
        function_name = node.func.id
        function = FORMULA_ALLOWED_FUNCTIONS[function_name]
        args = [self.visit(arg) for arg in node.args]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            return function(*args)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Formula contains unsupported syntax: {type(node).__name__}.")


def evaluate_compiled_formula(
    compiled: CompiledFormula,
    arrays_by_name: Dict[str, np.ndarray],
    formula_name: str,
) -> np.ndarray:
    missing = [name for name in compiled.referenced_names if name not in arrays_by_name]
    if missing:
        raise ValueError(f"Formula '{formula_name}' uses input map(s) that were not loaded: {', '.join(missing)}.")

    shapes = {arrays_by_name[name].shape for name in compiled.referenced_names}
    if len(shapes) != 1:
        details = ", ".join(f"{name}: {arrays_by_name[name].shape}" for name in compiled.referenced_names)
        raise ValueError(
            f"Formula '{formula_name}' cannot be calculated because referenced maps have different shapes: {details}."
        )
    target_shape = next(iter(shapes))

    evaluator = FormulaEvaluator(compiled, arrays_by_name)
    result = evaluator.visit(compiled.tree)

    if np.isscalar(result):
        result_array = np.full(target_shape, float(result), dtype=float)
    else:
        result_array = np.asarray(result, dtype=float)
        if result_array.shape == ():
            result_array = np.full(target_shape, float(result_array), dtype=float)
        elif result_array.shape != target_shape:
            raise ValueError(
                f"Formula '{formula_name}' produced shape {result_array.shape}, expected {target_shape}."
            )
        else:
            result_array = np.array(result_array, dtype=float, copy=True)

    result_array[~np.isfinite(result_array)] = np.nan
    return result_array


def create_sample_data_files(folder: str, seed: Optional[int] = DEFAULT_SAMPLE_SEED) -> Tuple[List[str], int]:
    """Create deterministic built-in demo CSV files.

    The demo is intentionally fixed: each click creates the same dataset. This
    keeps the demo simple and predictable. The generated CSV files include five
    metadata rows, matching the default preprocessing setting that removes rows
    [1-5].
    """
    rows = 384
    cols = 512
    used_seed = DEFAULT_SAMPLE_SEED if seed is None else int(seed)
    pixel_size_um = 0.1

    y = np.linspace(-1.0, 1.0, rows)
    x = np.linspace(-1.0, 1.0, cols)
    X, Y = np.meshgrid(x, y)

    rng = np.random.default_rng(used_seed)

    grayscale = (
        110
        + 30 * (X + 1.0)
        + 20 * (Y + 1.0)
        + 55 * np.exp(-((X + 0.45) ** 2 + (Y - 0.25) ** 2) / 0.030)
        + 75 * np.exp(-((X - 0.40) ** 2 + (Y + 0.10) ** 2) / 0.025)
        + 45 * np.exp(-((X + 0.05) ** 2 + (Y + 0.55) ** 2) / 0.018)
        + rng.normal(0, 4, (rows, cols))
    )

    dark_mask = (
        (((X + 0.65) ** 2 + (Y + 0.55) ** 2) < 0.06 ** 2)
        | (((X - 0.55) ** 2 + (Y - 0.50) ** 2) < 0.08 ** 2)
        | (((X - 0.05) ** 2 + (Y + 0.05) ** 2) < 0.07 ** 2)
    )
    grayscale[dark_mask] -= 95
    grayscale = np.clip(grayscale, 0, 255)

    element1 = (
        2.0
        + 0.55 * np.sin(2.4 * np.pi * (X + 1.0) / 2.0)
        + 0.35 * np.cos(1.8 * np.pi * (Y + 1.0) / 2.0)
        + 1.25 * np.exp(-((X + 0.30) ** 2 + (Y - 0.20) ** 2) / 0.045)
        + 0.90 * np.exp(-((X - 0.35) ** 2 + (Y + 0.35) ** 2) / 0.030)
        + rng.normal(0, 0.05, (rows, cols))
    )
    element1 = np.clip(element1 * 100.0, 0.05, None)

    element2 = (
        1.6
        + 0.45 * np.cos(2.0 * np.pi * (X + 1.0) / 2.0)
        + 0.40 * np.sin(2.2 * np.pi * (Y + 1.0) / 2.0)
        + 1.10 * np.exp(-((X - 0.10) ** 2 + (Y - 0.35) ** 2) / 0.050)
        + 0.75 * np.exp(-((X + 0.45) ** 2 + (Y + 0.10) ** 2) / 0.028)
        + rng.normal(0, 0.05, (rows, cols))
    )
    element2 = np.clip(element2 * 100.0, 0.05, None)

    os.makedirs(folder, exist_ok=True)

    grayscale_path = os.path.join(folder, "grayscale_demo.csv")
    element1_path = os.path.join(folder, "element1_demo.csv")
    element2_path = os.path.join(folder, "element2_demo.csv")

    def save_demo_csv(file_path: str, map_label: str, array: np.ndarray) -> None:
        metadata_rows = [
            ["Elementti demo data"],
            ["Map", map_label],
            ["Pixel size", f"{pixel_size_um:g} um/pixel"],
            ["Rows", str(rows)],
            ["Columns", str(cols)],
        ]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(metadata_rows)
            np.savetxt(f, array, delimiter=",", fmt="%.6f")

    save_demo_csv(grayscale_path, "Grey", grayscale)
    save_demo_csv(element1_path, "Element 1", element1)
    save_demo_csv(element2_path, "Element 2", element2)

    return [grayscale_path, element1_path, element2_path], used_seed

@dataclass
class FileReadSettings:
    manual_rows: str = "[1-5]"
    manual_columns: str = ""
    scale_factor: str = "0.01"
    pixel_size_um: str = ""


@dataclass
class OutputDisplaySettings:
    mode: str = "Continuous"
    colormap: str = "viridis"
    colorbar_side: str = "right"
    colorbar_label: str = ""
    display_min: str = ""
    display_max: str = ""
    bins: str = ""
    bin_colors: str = ""


@dataclass
class LineProfileCurveSettings:
    enabled: bool = True
    color: str = ""
    line_width: str = "2.0"


@dataclass
class LineProfileSettings:
    enabled: bool = False
    base_output_name: str = ""
    selected_output_names: List[str] = field(default_factory=list)
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    normalize_mode: str = "Raw values"
    image_title: str = ""
    image_x_label: str = ""
    image_y_label: str = ""
    title: str = ""
    # Blank x_label means automatic: physical distance if a pixel size is available, otherwise pixels.
    x_label: str = ""
    y_label: str = "Value"
    y_min: str = ""
    y_max: str = ""
    right_y_label: str = "Ratio value"
    right_y_min: str = ""
    right_y_max: str = ""
    legend_position: str = "Inside plot"
    curve_settings: Dict[str, LineProfileCurveSettings] = field(default_factory=dict)


DEFAULT_LINE_PROFILE_COLORS = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


@dataclass
class LowValueRule:
    enabled: bool = False
    floor: str = ""


@dataclass
class GrayscaleMaskRule:
    enabled: bool = True
    source_name: str = ""
    target_output_name: str = "All outputs"
    condition: str = "below"
    value_a: str = ""
    value_b: str = ""
    color: str = "white"


@dataclass
class ConditionalReplacementRule:
    enabled: bool = True
    condition_source_name: str = ""
    condition_operator: str = "above"
    value_a: str = ""
    value_b: str = ""
    target_name: str = ""
    replacement_value: str = ""


@dataclass
class MaskingAndNoiseSettings:
    enabled: bool = False
    grayscale_enabled: bool = False
    grayscale_rules: List[GrayscaleMaskRule] = field(default_factory=list)
    low_value_enabled: bool = False
    low_value_by_name: Dict[str, LowValueRule] = field(default_factory=dict)
    conditional_enabled: bool = False
    conditional_rules: List[ConditionalReplacementRule] = field(default_factory=list)


@dataclass
class ScaleBarSettings:
    enabled: bool = False
    length_um: str = ""
    color: str = "white"
    position: str = "lower right"


@dataclass
class FormulaOutput:
    name: str = ""
    expression: str = ""


@dataclass
class AppState:
    selected_file_paths: List[str] = field(default_factory=list)
    renamed_names_by_path: Dict[str, str] = field(default_factory=dict)
    selected_maps: List[str] = field(default_factory=list)
    selected_ratios: List[Tuple[str, str]] = field(default_factory=list)
    selected_formulas: List[FormulaOutput] = field(default_factory=list)
    file_processing_settings: Dict[str, FileReadSettings] = field(default_factory=dict)
    processed_min_max_by_name: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    display_settings: Dict[str, OutputDisplaySettings] = field(default_factory=dict)
    line_profile_settings: LineProfileSettings = field(default_factory=LineProfileSettings)
    masking_and_noise_settings: MaskingAndNoiseSettings = field(default_factory=MaskingAndNoiseSettings)
    scale_bar_settings: ScaleBarSettings = field(default_factory=ScaleBarSettings)
    output_folder: str = ""
    project_name: str = ""
    save_png: bool = True
    save_csv: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300
    sample_data_folder: str = ""
    sample_data_mode: str = ""
    sample_data_seed: Optional[int] = None

    def get_ordered_names(self) -> List[str]:
        return [
            self.renamed_names_by_path[path]
            for path in self.selected_file_paths
            if path in self.renamed_names_by_path
        ]

    def get_named_files(self) -> Dict[str, str]:
        return {
            self.renamed_names_by_path[path]: path
            for path in self.selected_file_paths
            if path in self.renamed_names_by_path
        }

    def get_output_names(self) -> List[str]:
        outputs = list(self.selected_maps)
        outputs.extend(f"{num} / {den}" for num, den in self.selected_ratios)
        outputs.extend(formula.name for formula in self.selected_formulas)
        return outputs

    def reset_outputs_and_display(self) -> None:
        self.selected_maps = []
        self.selected_ratios = []
        self.selected_formulas = []
        self.display_settings.clear()
        self.processed_min_max_by_name.clear()
        self.reset_line_profile_settings()

    def reset_line_profile_settings(self) -> None:
        self.line_profile_settings = LineProfileSettings()

    def reset_masking_settings(self) -> None:
        self.masking_and_noise_settings = MaskingAndNoiseSettings()

    def reset_scale_bar_settings(self) -> None:
        self.scale_bar_settings = ScaleBarSettings()

    def reset_all_dependent_settings(self) -> None:
        self.reset_outputs_and_display()
        self.reset_masking_settings()
        self.reset_scale_bar_settings()

    def clear_display_results_only(self) -> None:
        self.display_settings.clear()
        self.processed_min_max_by_name.clear()

    def prune_display_settings_to_current_outputs(self) -> None:
        allowed = set(self.get_output_names())
        self.display_settings = {
            name: settings
            for name, settings in self.display_settings.items()
            if name in allowed
        }
        self.processed_min_max_by_name = {
            name: minmax
            for name, minmax in self.processed_min_max_by_name.items()
            if name in allowed
        }
        self.prune_line_profile_settings_to_current_outputs()

    def prune_line_profile_settings_to_current_outputs(self) -> None:
        allowed_curves = set(self.get_output_names())
        allowed_base = allowed_curves | set(self.get_ordered_names())
        lp = self.line_profile_settings

        if lp.base_output_name not in allowed_base:
            lp.base_output_name = next(iter(allowed_base), "")
            lp.start_x = None
            lp.start_y = None
            lp.end_x = None
            lp.end_y = None

        lp.selected_output_names = [name for name in lp.selected_output_names if name in allowed_curves]
        lp.curve_settings = {
            name: settings
            for name, settings in lp.curve_settings.items()
            if name in allowed_curves
        }


class ProcessingEngine:
    def __init__(self, state: AppState):
        self.state = state

    def read_csv_cells(self, file_path: str) -> List[List[str]]:
        with open(file_path, "rb") as f:
            raw = f.read()

        raw = raw.replace(b"\xa0", b" ")
        text = raw.decode("utf-8", errors="ignore")

        reader = csv.reader(io.StringIO(text))
        rows = [[cell.strip() for cell in row] for row in reader]

        if not rows:
            raise ValueError(f"File '{os.path.basename(file_path)}' is empty.")

        max_len = max(len(row) for row in rows) if rows else 0
        if max_len == 0:
            raise ValueError(f"File '{os.path.basename(file_path)}' has no readable columns.")

        padded = [row + [""] * (max_len - len(row)) for row in rows]
        return padded

    def apply_manual_row_removal(self, rows: List[List[str]], manual_rows: List[int]) -> List[List[str]]:
        if not manual_rows:
            return rows

        manual_set = set(manual_rows)
        new_rows = [row for i, row in enumerate(rows) if i not in manual_set]

        if not new_rows:
            raise ValueError("All rows were removed. Please revise the rows to remove.")

        return new_rows

    def apply_manual_column_removal(self, rows: List[List[str]], manual_columns: List[int]) -> List[List[str]]:
        if not rows:
            return rows

        ncols = max(len(row) for row in rows)
        if not manual_columns:
            return rows

        manual_set = set(manual_columns)
        keep_indices = [j for j in range(ncols) if j not in manual_set]

        if not keep_indices:
            raise ValueError("All columns were removed. Please revise the columns to remove.")

        new_rows = []
        for row in rows:
            padded = row + [""] * (ncols - len(row))
            new_rows.append([padded[j] for j in keep_indices])

        return new_rows

    def convert_cells_to_numeric_array(self, rows: List[List[str]], file_label: str) -> np.ndarray:
        if not rows:
            raise ValueError(f"No rows remain in '{file_label}' after row removal.")

        ncols = max(len(row) for row in rows)
        if ncols == 0:
            raise ValueError(f"No columns remain in '{file_label}' after column removal.")

        numeric_rows = []
        for r, row in enumerate(rows, start=1):
            padded = row + [""] * (ncols - len(row))
            numeric_row = []
            for c, cell in enumerate(padded, start=1):
                text = cell.strip()
                if text == "":
                    numeric_row.append(np.nan)
                    continue
                try:
                    numeric_row.append(float(text))
                except ValueError:
                    raise ValueError(
                        f"After removing the chosen rows/columns, '{file_label}' still contains a non-numeric value "
                        f"at processed row {r}, column {c}: '{text}'.\n\n"
                        f"Please revise the rows or columns to remove."
                    )
            numeric_rows.append(numeric_row)

        arr = np.array(numeric_rows, dtype=float)
        if arr.size == 0:
            raise ValueError(f"No numeric data remain in '{file_label}'.")
        return arr

    def load_processed_array(self, file_path: str, settings: FileReadSettings) -> np.ndarray:
        file_label = os.path.basename(file_path)
        rows = self.read_csv_cells(file_path)

        manual_rows = parse_manual_index_list(settings.manual_rows, "Rows to remove")
        manual_columns = parse_manual_index_list(settings.manual_columns, "Columns to remove")

        rows = self.apply_manual_row_removal(rows, manual_rows)
        rows = self.apply_manual_column_removal(rows, manual_columns)

        arr = self.convert_cells_to_numeric_array(rows, file_label)
        arr = arr * parse_float_user_text(settings.scale_factor, "Scale factor")
        return arr

    def load_raw_arrays(self) -> Dict[str, np.ndarray]:
        arrays = {}
        for name, path in self.state.get_named_files().items():
            settings = self.state.file_processing_settings[path]
            arrays[name] = self.load_processed_array(path, settings)
        return arrays

    def apply_low_value_protection(self, arrays_by_name: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        settings = self.state.masking_and_noise_settings
        output = {name: np.array(arr, copy=True) for name, arr in arrays_by_name.items()}

        if not settings.enabled or not settings.low_value_enabled:
            return output

        for name, arr in output.items():
            rule = settings.low_value_by_name.get(name)
            if not rule or not rule.enabled:
                continue

            floor_text = str(rule.floor).strip()
            if floor_text == "":
                continue

            floor_value = parse_float_user_text(floor_text, "Low-value floor")
            finite_mask = np.isfinite(arr)
            arr[finite_mask & (arr <= floor_value)] = floor_value

        return output

    def apply_conditional_replacements(self, arrays_by_name: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        settings = self.state.masking_and_noise_settings
        output = {name: np.array(arr, copy=True) for name, arr in arrays_by_name.items()}

        if not settings.enabled or not settings.conditional_enabled:
            return output

        for idx, rule in enumerate(settings.conditional_rules, start=1):
            if not rule.enabled:
                continue

            source_name = rule.condition_source_name.strip()
            target_name = rule.target_name.strip()
            operator = normalize_conditional_operator(rule.condition_operator)
            replacement_text = rule.replacement_value.strip()

            if source_name == "" or target_name == "" or replacement_text == "":
                raise ValueError(f"Conditional replacement rule {idx} is incomplete.")

            if source_name not in output:
                raise ValueError(f"Conditional replacement rule {idx} uses unknown source file '{source_name}'.")
            if target_name not in output:
                raise ValueError(f"Conditional replacement rule {idx} uses unknown target file '{target_name}'.")

            source_arr = output[source_name]
            target_arr = output[target_name]

            if source_arr.shape != target_arr.shape:
                raise ValueError(
                    f"Conditional replacement rule {idx} cannot be applied because "
                    f"'{source_name}' and '{target_name}' have different shapes: "
                    f"{source_arr.shape} and {target_arr.shape}."
                )

            replacement_value = parse_float_user_text(replacement_text, "Replacement value")

            if operator == "between":
                a_text = rule.value_a.strip()
                b_text = rule.value_b.strip()
                if a_text == "" or b_text == "":
                    raise ValueError(f"Conditional replacement rule {idx} requires both Value A and Value B.")
                a = parse_float_user_text(a_text, "Value A")
                b = parse_float_user_text(b_text, "Value B")
                low = min(a, b)
                high = max(a, b)
                condition_mask = np.isfinite(source_arr) & (source_arr >= low) & (source_arr <= high)
            else:
                a_text = rule.value_a.strip()
                if a_text == "":
                    raise ValueError(f"Conditional replacement rule {idx} requires Value A.")
                a = parse_float_user_text(a_text, "Value A")

                if operator == "above":
                    condition_mask = np.isfinite(source_arr) & (source_arr >= a)
                elif operator == "below":
                    condition_mask = np.isfinite(source_arr) & (source_arr <= a)
                else:
                    raise ValueError(f"Conditional replacement rule {idx} uses unknown operator '{operator}'.")

            target_arr[condition_mask] = replacement_value

        return output

    def apply_processing_rules(self, raw_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        working = self.apply_low_value_protection(raw_arrays)
        working = self.apply_conditional_replacements(working)
        return working

    @staticmethod
    def grayscale_rule_applies_to_output(rule: GrayscaleMaskRule, output_name: Optional[str]) -> bool:
        target = rule.target_output_name.strip()
        if output_name is None:
            return True
        return target == "" or target == "All outputs" or target == output_name

    def build_grayscale_rule_mask(
        self,
        raw_arrays_by_name: Dict[str, np.ndarray],
        rule: GrayscaleMaskRule,
        rule_index: int,
    ) -> np.ndarray:
        source_name = rule.source_name.strip()
        condition = normalize_grayscale_condition(rule.condition)

        if source_name == "":
            raise ValueError(f"Grayscale mask rule {rule_index} has no source file.")
        if source_name not in raw_arrays_by_name:
            raise ValueError(f"Grayscale mask rule {rule_index} uses unknown source file '{source_name}'.")

        source_array = raw_arrays_by_name[source_name]

        if condition in ("below", "above"):
            if rule.value_a.strip() == "":
                raise ValueError(f"Grayscale mask rule {rule_index} requires Value A.")
            a = parse_float_user_text(rule.value_a, "Grayscale mask Value A")

            if condition == "below":
                return np.isfinite(source_array) & (source_array <= a)
            return np.isfinite(source_array) & (source_array >= a)

        if condition in ("between", "outside"):
            if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                raise ValueError(f"Grayscale mask rule {rule_index} requires both Value A and Value B.")
            a = parse_float_user_text(rule.value_a, "Grayscale mask Value A")
            b = parse_float_user_text(rule.value_b, "Grayscale mask Value B")
            low = min(a, b)
            high = max(a, b)

            if condition == "between":
                return np.isfinite(source_array) & (source_array >= low) & (source_array <= high)

            return np.isfinite(source_array) & ((source_array < low) | (source_array > high))

        raise ValueError(f"Grayscale mask rule {rule_index} uses unknown condition '{condition}'.")

    def build_combined_grayscale_mask(
        self,
        raw_arrays_by_name: Dict[str, np.ndarray],
        output_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        settings = self.state.masking_and_noise_settings

        if not settings.enabled or not settings.grayscale_enabled:
            return None

        active_rules = [
            rule
            for rule in settings.grayscale_rules
            if rule.enabled and self.grayscale_rule_applies_to_output(rule, output_name)
        ]
        if not active_rules:
            return None

        combined_mask = None

        for idx, rule in enumerate(active_rules, start=1):
            rule_mask = self.build_grayscale_rule_mask(raw_arrays_by_name, rule, idx)

            if combined_mask is None:
                combined_mask = np.array(rule_mask, copy=True)
            else:
                if rule_mask.shape != combined_mask.shape:
                    raise ValueError(
                        f"Grayscale mask rule {idx} has shape {rule_mask.shape}, "
                        f"but previous grayscale mask rules use shape {combined_mask.shape}."
                    )
                combined_mask |= rule_mask

        return combined_mask

    def build_grayscale_mask_layers(
        self,
        raw_arrays_by_name: Dict[str, np.ndarray],
        output_name: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, str]]:
        settings = self.state.masking_and_noise_settings

        if not settings.enabled or not settings.grayscale_enabled:
            return []

        active_rules = [
            rule
            for rule in settings.grayscale_rules
            if rule.enabled and self.grayscale_rule_applies_to_output(rule, output_name)
        ]

        layers = []
        for idx, rule in enumerate(active_rules, start=1):
            rule_mask = self.build_grayscale_rule_mask(raw_arrays_by_name, rule, idx)
            layers.append((rule_mask, rule.color.strip() or "white"))

        return layers

    @staticmethod
    def get_visible_finite_values(array: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is not None and mask.shape != array.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {array.shape}.")

        visible = array if mask is None else array[~mask]
        return visible[np.isfinite(visible)]

    def calculate_ratio_array(
        self,
        working_arrays: Dict[str, np.ndarray],
        numerator: str,
        denominator: str,
    ) -> np.ndarray:
        if numerator not in working_arrays:
            raise ValueError(f"Ratio numerator '{numerator}' was not loaded.")
        if denominator not in working_arrays:
            raise ValueError(f"Ratio denominator '{denominator}' was not loaded.")

        num_arr = working_arrays[numerator]
        den_arr = working_arrays[denominator]

        if num_arr.shape != den_arr.shape:
            raise ValueError(
                f"Ratio '{numerator} / {denominator}' cannot be calculated because the two arrays "
                f"have different shapes: {num_arr.shape} and {den_arr.shape}."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_arr = num_arr / den_arr
            ratio_arr[~np.isfinite(ratio_arr)] = np.nan

        return ratio_arr

    def compile_formula(self, expression: str) -> CompiledFormula:
        return compile_formula_expression(expression, self.state.get_ordered_names())

    def get_formula_referenced_names(self, expression: str) -> List[str]:
        return self.compile_formula(expression).referenced_names

    def calculate_formula_array(
        self,
        working_arrays: Dict[str, np.ndarray],
        formula_name: str,
        expression: str,
    ) -> np.ndarray:
        compiled = self.compile_formula(expression)
        return evaluate_compiled_formula(compiled, working_arrays, formula_name)

    def build_output_arrays(
        self,
        raw_arrays: Optional[Dict[str, np.ndarray]] = None,
        working_arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if raw_arrays is None:
            raw_arrays = self.load_raw_arrays()
        if working_arrays is None:
            working_arrays = self.apply_processing_rules(raw_arrays)

        output_arrays: Dict[str, np.ndarray] = {}

        for map_name in self.state.selected_maps:
            if map_name not in working_arrays:
                raise ValueError(f"Selected map '{map_name}' was not loaded.")
            output_arrays[map_name] = working_arrays[map_name]

        for numerator, denominator in self.state.selected_ratios:
            ratio_name = f"{numerator} / {denominator}"
            output_arrays[ratio_name] = self.calculate_ratio_array(
                working_arrays,
                numerator,
                denominator,
            )

        for formula in self.state.selected_formulas:
            formula_name = formula.name.strip()
            if formula_name == "":
                continue
            output_arrays[formula_name] = self.calculate_formula_array(
                working_arrays,
                formula_name,
                formula.expression,
            )

        return output_arrays

    @staticmethod
    def choose_line_profile_distance_axis(distance_pixels: np.ndarray, pixel_size_um: Optional[float]) -> Tuple[np.ndarray, str, str, Optional[float]]:
        """Return distances for plotting and a suitable physical unit when possible."""
        pixel_distances = np.asarray(distance_pixels, dtype=float)
        if pixel_size_um is None or pixel_size_um <= 0:
            return pixel_distances, "pixels", "Distance along line (pixels)", None

        distance_um = pixel_distances * float(pixel_size_um)
        finite = distance_um[np.isfinite(distance_um)]
        total_um = float(np.max(finite)) if finite.size else 0.0

        if total_um > 0 and total_um < 1.0:
            return distance_um * 1000.0, "nm", "Distance along line (nm)", float(pixel_size_um)
        if total_um >= 1000.0:
            return distance_um / 1000.0, "mm", "Distance along line (mm)", float(pixel_size_um)
        return distance_um, "µm", "Distance along line (µm)", float(pixel_size_um)

    def formula_expression_uses_division(self, expression: str) -> bool:
        """Return True for formula maps that are ratio-like because they use division."""
        try:
            compiled = self.compile_formula(expression)
            return any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(compiled.tree))
        except Exception:
            return "/" in str(expression)

    @staticmethod
    def formula_tree_contains_division(compiled: CompiledFormula) -> bool:
        return any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(compiled.tree))

    def is_ratio_like_output(self, output_name: str) -> bool:
        """Return True for normal ratios and formula maps that contain division."""
        ratio_names = {f"{num} / {den}" for num, den in self.state.selected_ratios}
        if output_name in ratio_names:
            return True

        for formula in self.state.selected_formulas:
            formula_name = formula.name.strip()
            if formula_name != output_name:
                continue
            try:
                compiled = self.compile_formula(formula.expression)
                return self.formula_tree_contains_division(compiled)
            except Exception:
                # If a formula somehow cannot be compiled here, keep it on the
                # value axis so the plotting code does not guess that it is a ratio.
                return False

        return False

    def classify_line_profile_outputs(self, selected_output_names: List[str]) -> Tuple[List[str], List[str]]:
        """Split selected outputs into normal value curves and ratio-like curves."""
        value_curves = []
        ratio_curves = []
        for name in selected_output_names:
            if self.is_ratio_like_output(name):
                ratio_curves.append(name)
            else:
                value_curves.append(name)
        return value_curves, ratio_curves

    @staticmethod
    def apply_manual_y_limits(ax, min_text: str, max_text: str, axis_label: str) -> bool:
        """Apply optional manual y-axis limits. Returns True when either limit is manual."""
        min_text = str(min_text).strip()
        max_text = str(max_text).strip()
        if not min_text and not max_text:
            return False

        current_bottom, current_top = ax.get_ylim()
        bottom = parse_float_user_text(min_text, f"{axis_label} minimum") if min_text else current_bottom
        top = parse_float_user_text(max_text, f"{axis_label} maximum") if max_text else current_top
        if bottom >= top:
            raise ValueError(f"{axis_label} minimum must be smaller than {axis_label} maximum.")
        ax.set_ylim(bottom, top)
        return True


    @staticmethod
    def normalize_line_profile_values(values: np.ndarray, normalize_mode: str) -> np.ndarray:
        mode = normalize_mode.strip().lower()
        normalized = np.array(values, dtype=float, copy=True)
        finite = normalized[np.isfinite(normalized)]

        if finite.size == 0 or mode == "raw values":
            return normalized

        if mode == "normalize 0-1":
            min_value = float(np.min(finite))
            max_value = float(np.max(finite))
            if max_value == min_value:
                normalized[np.isfinite(normalized)] = 0.0
            else:
                normalized = (normalized - min_value) / (max_value - min_value)
            return normalized

        if mode == "normalize by maximum":
            max_value = float(np.max(finite))
            if max_value != 0:
                normalized = normalized / max_value
            return normalized

        return normalized

    def extract_line_profile_data(
        self,
        output_arrays: Dict[str, np.ndarray],
        base_output_name: str,
        selected_output_names: List[str],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        normalize_mode: str = "Raw values",
    ) -> dict:
        if base_output_name not in output_arrays:
            raise ValueError(f"Line-profile base output '{base_output_name}' is not available.")
        if not selected_output_names:
            raise ValueError("Please select at least one output to plot in the line profile.")

        base_array = output_arrays[base_output_name]
        rows, cols = base_array.shape

        for output_name in selected_output_names:
            if output_name not in output_arrays:
                raise ValueError(f"Line-profile output '{output_name}' is not available.")
            if output_arrays[output_name].shape != base_array.shape:
                raise ValueError(
                    f"Line-profile output '{output_name}' has shape {output_arrays[output_name].shape}, "
                    f"but the selected base output '{base_output_name}' has shape {base_array.shape}. "
                    "Line profiles require outputs with the same pixel dimensions."
                )

        dx = float(end_x) - float(start_x)
        dy = float(end_y) - float(start_y)
        line_length = float(np.hypot(dx, dy))
        if line_length <= 0:
            raise ValueError("The line-profile start and end points are identical. Please draw a longer line.")

        n_points = max(int(np.ceil(line_length)) + 1, 2)
        xs = np.linspace(float(start_x), float(end_x), n_points)
        ys = np.linspace(float(start_y), float(end_y), n_points)

        xi = np.clip(np.rint(xs).astype(int), 0, cols - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, rows - 1)
        distances = np.sqrt((xs - float(start_x)) ** 2 + (ys - float(start_y)) ** 2)

        pixel_size_um = None
        try:
            pixel_size_um = self.get_pixel_size_um_for_output(base_output_name)
        except Exception:
            # If the base output has no reliable common pixel size, keep pixel
            # distances instead of guessing a physical scale.
            pixel_size_um = None
        distance_axis, distance_axis_unit, distance_axis_label, used_pixel_size_um = self.choose_line_profile_distance_axis(
            distances,
            pixel_size_um,
        )

        values_by_name = {}
        raw_values_by_name = {}
        for output_name in selected_output_names:
            raw_values = output_arrays[output_name][yi, xi].astype(float)
            raw_values_by_name[output_name] = raw_values
            values_by_name[output_name] = self.normalize_line_profile_values(raw_values, normalize_mode)

        return {
            "base_output_name": base_output_name,
            "selected_output_names": list(selected_output_names),
            "start_x": float(start_x),
            "start_y": float(start_y),
            "end_x": float(end_x),
            "end_y": float(end_y),
            "distance_pixels": distances,
            "distance_axis": distance_axis,
            "distance_axis_unit": distance_axis_unit,
            "distance_axis_label": distance_axis_label,
            "pixel_size_um_per_pixel": used_pixel_size_um,
            "x_pixels": xs,
            "y_pixels": ys,
            "values_by_name": values_by_name,
            "raw_values_by_name": raw_values_by_name,
            "normalize_mode": normalize_mode,
        }

    def save_line_profile_csv(self, profile_data: dict, save_path: str) -> None:
        selected_output_names = profile_data["selected_output_names"]
        normalize_mode = profile_data.get("normalize_mode", "Raw values")
        include_raw_columns = normalize_mode.strip().lower() != "raw values"
        distance_unit = profile_data.get("distance_axis_unit", "pixels")
        distance_axis = profile_data.get("distance_axis", profile_data["distance_pixels"])
        safe_unit = str(distance_unit).replace("µ", "u")

        header = ["distance_pixels"]
        if distance_unit != "pixels":
            header.append(f"distance_{safe_unit}")
        header.extend(["x_pixels", "y_pixels"])
        for output_name in selected_output_names:
            header.append(f"{output_name}_plotted")
            if include_raw_columns:
                header.append(f"{output_name}_raw")

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["normalization_mode", normalize_mode])
            writer.writerow(["distance_axis", profile_data.get("distance_axis_label", "Distance along line (pixels)")])
            if profile_data.get("pixel_size_um_per_pixel") is not None:
                writer.writerow(["pixel_size_um_per_pixel", profile_data.get("pixel_size_um_per_pixel")])
            writer.writerow(header)

            distances = profile_data["distance_pixels"]
            xs = profile_data["x_pixels"]
            ys = profile_data["y_pixels"]
            values_by_name = profile_data["values_by_name"]
            raw_values_by_name = profile_data["raw_values_by_name"]

            for i in range(len(distances)):
                row = [distances[i]]
                if distance_unit != "pixels":
                    row.append(distance_axis[i])
                row.extend([xs[i], ys[i]])
                for output_name in selected_output_names:
                    row.append(values_by_name[output_name][i])
                    if include_raw_columns:
                        row.append(raw_values_by_name[output_name][i])
                writer.writerow(row)

    def plot_line_profile_axis(self, ax, profile_data: dict, title_override: Optional[str] = None) -> None:
        settings = self.state.line_profile_settings
        selected_output_names = profile_data["selected_output_names"]
        distances = np.asarray(profile_data.get("distance_axis", profile_data["distance_pixels"]), dtype=float)
        values_by_name = profile_data["values_by_name"]
        normalize_mode = profile_data.get("normalize_mode", "Raw values")
        raw_mode = normalize_mode.strip().lower() == "raw values"

        value_names, ratio_names = self.classify_line_profile_outputs(selected_output_names)
        use_right_axis = bool(value_names) and bool(ratio_names)

        if use_right_axis:
            left_names = value_names
            right_names = ratio_names
            right_ax = ax.twinx()
            try:
                right_ax.patch.set_visible(False)
            except Exception:
                pass
        else:
            left_names = list(selected_output_names)
            right_names = []
            right_ax = None

        try:
            ax._elementti_right_axis = right_ax
        except Exception:
            pass

        all_handles = []
        all_labels = []

        def plot_curve(target_ax, output_name: str):
            curve_settings = settings.curve_settings.get(output_name, LineProfileCurveSettings())
            color = curve_settings.color.strip()
            line_width_text = curve_settings.line_width.strip()

            try:
                line_width = parse_float_user_text(line_width_text, "Line width") if line_width_text else 2.0
            except ValueError:
                line_width = 2.0

            plot_kwargs = {"linewidth": line_width, "label": output_name}
            if color and is_color_like(color):
                plot_kwargs["color"] = color

            line, = target_ax.plot(distances, values_by_name[output_name], **plot_kwargs)
            all_handles.append(line)
            all_labels.append(output_name)

        for output_name in left_names:
            plot_curve(ax, output_name)
        if right_ax is not None:
            for output_name in right_names:
                plot_curve(right_ax, output_name)

        auto_x_label = profile_data.get("distance_axis_label", "Distance along line (pixels)")
        custom_x_label = settings.x_label.strip()
        if custom_x_label and custom_x_label != "Distance along line (pixels)":
            x_label = custom_x_label
        else:
            x_label = auto_x_label
        ax.set_xlabel(x_label, fontname=DEFAULT_FONT_NAME)

        custom_left_y_label = settings.y_label.strip()
        left_contains_only_ratio_like = bool(ratio_names) and not value_names and set(left_names) == set(ratio_names)
        if custom_left_y_label and custom_left_y_label != "Value":
            left_y_label = custom_left_y_label
        elif not raw_mode:
            if normalize_mode.strip().lower() == "normalize 0-1":
                left_y_label = "Normalized value (0-1)"
            elif normalize_mode.strip().lower() == "normalize by maximum":
                left_y_label = "Value / maximum"
            else:
                left_y_label = "Value"
        elif left_contains_only_ratio_like:
            left_y_label = settings.right_y_label.strip() or "Ratio value"
        else:
            left_y_label = "Value"
        ax.set_ylabel(left_y_label, fontname=DEFAULT_FONT_NAME)

        if right_ax is not None:
            custom_right_y_label = settings.right_y_label.strip()
            if custom_right_y_label and custom_right_y_label != "Ratio value":
                right_y_label = custom_right_y_label
            elif not raw_mode:
                if normalize_mode.strip().lower() == "normalize 0-1":
                    right_y_label = "Normalized ratio (0-1)"
                elif normalize_mode.strip().lower() == "normalize by maximum":
                    right_y_label = "Ratio / maximum"
                else:
                    right_y_label = "Ratio value"
            else:
                right_y_label = "Ratio value"
            right_ax.set_ylabel(right_y_label, fontname=DEFAULT_FONT_NAME)

        plot_title = title_override if title_override is not None else settings.title.strip()
        if plot_title:
            ax.set_title(plot_title, fontname=DEFAULT_FONT_NAME)

        legend_position = normalize_line_profile_legend_position(settings.legend_position)
        legend_axis = right_ax if right_ax is not None else ax
        if legend_position == "Outside right":
            add_outside_right_legend(
                legend_axis,
                all_handles,
                all_labels,
                right_axis_present=right_ax is not None,
            )
        else:
            # Reserve legend headroom on all relevant y-axes, but attach the
            # combined legend to the topmost axis. For dual-y-axis plots this
            # keeps right-axis curves from being drawn over the legend.
            apply_inside_top_legend_with_headroom(
                legend_axis,
                legend_handles=all_handles,
                legend_labels=all_labels,
                place_legend=True,
                extra_axes=[ax] if right_ax is not None else None,
            )

        if left_contains_only_ratio_like:
            self.apply_manual_y_limits(ax, settings.right_y_min, settings.right_y_max, "Ratio y-axis")
            set_y_axis_endpoint_labels_for_manual_limits(ax, settings.right_y_min, settings.right_y_max)
        else:
            self.apply_manual_y_limits(ax, settings.y_min, settings.y_max, "Left y-axis")
            set_y_axis_endpoint_labels_for_manual_limits(ax, settings.y_min, settings.y_max)
        if right_ax is not None:
            self.apply_manual_y_limits(right_ax, settings.right_y_min, settings.right_y_max, "Right y-axis")
            set_y_axis_endpoint_labels_for_manual_limits(right_ax, settings.right_y_min, settings.right_y_max)

        ax.grid(True, alpha=0.25)
        if right_ax is not None:
            right_ax.grid(False)

    @staticmethod
    def format_scale_bar_number(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:g}"

    @staticmethod
    def generate_nice_scale_lengths_um() -> List[float]:
        # Candidate physical lengths are intentionally restricted to clean
        # publication-style values. Lengths are stored internally in µm, but
        # labels may be shown as nm, µm, or mm.
        mantissas = [1, 2, 5, 10, 15, 20, 30, 50, 100]
        candidates = set()
        for exponent in range(-6, 10):
            factor = 10 ** exponent
            for mantissa in mantissas:
                candidates.add(float(mantissa * factor))
        return sorted(c for c in candidates if c > 0)

    @classmethod
    def choose_nice_scale_bar_length_um(cls, image_width_um: float, pixel_size_um: float) -> float:
        if image_width_um <= 0 or pixel_size_um <= 0:
            return 1.0

        # Aim for a clean value close to one fifth to one fourth of the image width.
        target_um = image_width_um * 0.225
        preferred_low_um = image_width_um * 0.18
        preferred_high_um = image_width_um * 0.25
        visible_low_um = max(pixel_size_um * 10.0, image_width_um * 0.05)
        visible_high_um = image_width_um * 0.50

        if visible_low_um > visible_high_um:
            return image_width_um * 0.25

        candidates = [
            value for value in cls.generate_nice_scale_lengths_um()
            if visible_low_um <= value <= visible_high_um
        ]
        if not candidates:
            # Very small or unusual images: use a safe value near the target.
            return min(max(target_um, visible_low_um), visible_high_um)

        preferred = [value for value in candidates if preferred_low_um <= value <= preferred_high_um]
        pool = preferred if preferred else candidates
        return float(min(pool, key=lambda value: abs(value - target_um)))

    @staticmethod
    def format_scale_bar_label(length_um: float) -> str:
        if length_um < 1.0:
            length_nm = length_um * 1000.0
            return f"{ProcessingEngine.format_scale_bar_number(length_nm)} nm"
        if length_um >= 1000.0:
            length_mm = length_um / 1000.0
            return f"{ProcessingEngine.format_scale_bar_number(length_mm)} mm"
        return f"{ProcessingEngine.format_scale_bar_number(length_um)} µm"

    @staticmethod
    def format_line_length_number(value: float, allow_decimal: bool = False) -> str:
        if allow_decimal:
            rounded = round(float(value), 1)
            if abs(rounded - round(rounded)) < 1e-9:
                return str(int(round(rounded)))
            return f"{rounded:.1f}"
        return str(int(round(float(value))))

    @staticmethod
    def format_line_length_label(length_um: float) -> str:
        if length_um < 1.0:
            value_nm = max(0.1, length_um * 1000.0)
            return f"{ProcessingEngine.format_line_length_number(value_nm, allow_decimal=value_nm < 10.0)} nm"
        if length_um >= 1000.0:
            value_mm = max(0.1, length_um / 1000.0)
            return f"{ProcessingEngine.format_line_length_number(value_mm, allow_decimal=True)} mm"
        allow_decimal = length_um < 10.0
        return f"{ProcessingEngine.format_line_length_number(max(0.1, length_um), allow_decimal=allow_decimal)} µm"

    def get_pixel_size_um_for_input_name(self, input_name: str) -> Optional[float]:
        for path, name in self.state.renamed_names_by_path.items():
            if name != input_name:
                continue
            settings = self.state.file_processing_settings.get(path)
            if not settings:
                return None
            pixel_size_text = settings.pixel_size_um.strip()
            if pixel_size_text == "":
                return None
            pixel_size_um = parse_pixel_size_text_to_um(pixel_size_text, "Pixel size")
            if pixel_size_um <= 0:
                raise ValueError(f"Pixel size for '{input_name}' must be greater than zero.")
            return pixel_size_um
        return None

    def get_pixel_size_um_for_output(self, output_name: str) -> Optional[float]:
        direct_value = self.get_pixel_size_um_for_input_name(output_name)
        if direct_value is not None:
            return direct_value

        for numerator, denominator in self.state.selected_ratios:
            ratio_name = f"{numerator} / {denominator}"
            if ratio_name != output_name:
                continue

            numerator_value = self.get_pixel_size_um_for_input_name(numerator)
            denominator_value = self.get_pixel_size_um_for_input_name(denominator)
            if numerator_value is None or denominator_value is None:
                return None

            tolerance = max(abs(numerator_value), abs(denominator_value), 1.0) * 1e-6
            if abs(numerator_value - denominator_value) > tolerance:
                raise ValueError(
                    f"Scale bar cannot be assigned to ratio '{output_name}' because '{numerator}' "
                    f"and '{denominator}' have different pixel sizes "
                    f"({numerator_value:g} and {denominator_value:g} µm/pixel)."
                )
            return numerator_value

        for formula in self.state.selected_formulas:
            formula_name = formula.name.strip()
            if formula_name != output_name:
                continue

            referenced_names = self.get_formula_referenced_names(formula.expression)
            pixel_sizes = []
            missing_names = []
            for input_name in referenced_names:
                value = self.get_pixel_size_um_for_input_name(input_name)
                if value is None:
                    missing_names.append(input_name)
                else:
                    pixel_sizes.append((input_name, value))

            if missing_names:
                return None
            if not pixel_sizes:
                return None

            first_name, first_value = pixel_sizes[0]
            for other_name, other_value in pixel_sizes[1:]:
                tolerance = max(abs(first_value), abs(other_value), 1.0) * 1e-6
                if abs(first_value - other_value) > tolerance:
                    details = ", ".join(f"{name}: {value:g}" for name, value in pixel_sizes)
                    raise ValueError(
                        f"Scale bar cannot be assigned to formula '{output_name}' because referenced input maps "
                        f"have different pixel sizes (µm/pixel): {details}."
                    )
            return first_value

        return None

    def get_scale_bar_draw_info(self, array_shape: Tuple[int, int], pixel_size_um: Optional[float]) -> Optional[dict]:
        settings = self.state.scale_bar_settings
        if not settings.enabled:
            return None

        if pixel_size_um is None:
            raise ValueError("Scale-bar export is enabled, but no pixel size is available for this output.")
        if pixel_size_um <= 0:
            raise ValueError("Pixel size (µm/pixel) must be greater than zero.")

        rows, cols = array_shape
        manual_length_text = settings.length_um.strip()
        if manual_length_text:
            length_um = parse_float_user_text(manual_length_text, "Scale-bar length (µm)")
            if length_um <= 0:
                raise ValueError("Scale-bar length (µm) must be greater than zero.")
            length_px = length_um / pixel_size_um
            if length_px < 10:
                raise ValueError(
                    "The selected scale-bar length is too small to be visible "
                    f"({length_px:.1f} pixels). Choose a longer value or leave the length blank for automatic selection."
                )
            if length_px > cols * 0.50:
                raise ValueError(
                    "The selected scale-bar length is too large for this image "
                    f"({length_px:.1f} pixels, image width {cols} pixels). Choose a shorter value or leave the length blank for automatic selection."
                )
        else:
            image_width_um = cols * pixel_size_um
            length_um = self.choose_nice_scale_bar_length_um(image_width_um, pixel_size_um)
            length_px = length_um / pixel_size_um

        return {
            "length_px": float(length_px),
            "length_um": float(length_um),
            "color": settings.color.strip() or "white",
            "position": settings.position.strip().lower() or "lower right",
        }

    def draw_scale_bar(self, ax, array_shape: Tuple[int, int], pixel_size_um: Optional[float]) -> None:
        info = self.get_scale_bar_draw_info(array_shape, pixel_size_um)
        if info is None:
            return

        rows, cols = array_shape
        length_px = info["length_px"]
        color = info["color"]
        position = info["position"]
        label = self.format_scale_bar_label(info["length_um"])

        margin_x = max(cols * 0.05, 8.0)
        margin_y = max(rows * 0.06, 8.0)
        label_offset = max(rows * 0.045, 6.0)
        line_width = max(2.5, min(rows, cols) * 0.012)

        upper = position.startswith("upper")
        right = position.endswith("right")

        x_end = cols - margin_x if right else margin_x + length_px
        x_start = x_end - length_px if right else x_end - length_px
        y = margin_y if upper else rows - margin_y

        if not right:
            x_start = margin_x
            x_end = x_start + length_px

        ax.plot([x_start, x_end], [y, y], color=color, linewidth=line_width, solid_capstyle="butt")
        text_y = y + label_offset if upper else y - label_offset
        va = "top" if upper else "bottom"

        try:
            r, g, b = to_rgb(color)
            brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
        except Exception:
            brightness = 1.0
        box_facecolor = "black" if brightness > 0.45 else "white"
        box_alpha = 0.48

        ax.text(
            (x_start + x_end) / 2.0,
            text_y,
            label,
            color=color,
            ha="center",
            va=va,
            fontsize=10,
            fontname=DEFAULT_FONT_NAME,
            bbox=dict(facecolor=box_facecolor, alpha=box_alpha, edgecolor="none", pad=1.6),
        )

    def draw_line_length_label(
        self,
        ax,
        line_coords: Optional[Tuple[float, float, float, float]],
        pixel_size_um: Optional[float],
        array_shape: Tuple[int, int],
    ) -> None:
        if line_coords is None or pixel_size_um is None or pixel_size_um <= 0:
            return

        start_x, start_y, end_x, end_y = line_coords
        length_px = float(np.hypot(float(end_x) - float(start_x), float(end_y) - float(start_y)))
        if length_px <= 0:
            return

        label = self.format_line_length_label(length_px * pixel_size_um)
        rows, cols = array_shape
        mid_x = (float(start_x) + float(end_x)) / 2.0
        mid_y = (float(start_y) + float(end_y)) / 2.0
        offset_y = max(rows * 0.025, 5.0)
        text_y = float(np.clip(mid_y - offset_y, 0, rows - 1))

        ax.text(
            mid_x,
            text_y,
            label,
            color="white",
            ha="center",
            va="bottom",
            fontsize=9,
            fontname=DEFAULT_FONT_NAME,
            bbox=dict(facecolor="black", alpha=0.62, edgecolor="none", pad=1.5),
            zorder=30,
        )

    def plot_output_map_axis(
        self,
        ax,
        array: np.ndarray,
        output_name: str,
        settings: OutputDisplaySettings,
        mask_layers: Optional[List[Tuple[np.ndarray, str]]] = None,
        line_coords: Optional[Tuple[float, float, float, float]] = None,
        show_colorbar: bool = True,
        add_scale_bar: bool = False,
        pixel_size_um: Optional[float] = None,
        colorbar_axis=None,
        colorbar_side_override: Optional[str] = None,
        colorbar_label_override: Optional[str] = None,
        show_line_length_label: bool = False,
    ) -> None:
        combined_mask = None

        if mask_layers:
            for idx, (rule_mask, _rule_color) in enumerate(mask_layers, start=1):
                if rule_mask.shape != array.shape:
                    raise ValueError(
                        f"Grayscale mask layer {idx} has shape {rule_mask.shape}, "
                        f"but output '{output_name}' has shape {array.shape}."
                    )
                if combined_mask is None:
                    combined_mask = np.array(rule_mask, copy=True)
                else:
                    combined_mask |= rule_mask

        finite = self.get_visible_finite_values(array, mask=combined_mask)
        if finite.size == 0:
            raise ValueError(f"No visible finite values found for '{output_name}'.")

        actual_min = float(np.min(finite))
        actual_max = float(np.max(finite))
        manual_display_min = settings.display_min.strip() != ""
        manual_display_max = settings.display_max.strip() != ""
        vmin = parse_float_user_text(settings.display_min, "Display minimum") if manual_display_min else actual_min
        vmax = parse_float_user_text(settings.display_max, "Display maximum") if manual_display_max else actual_max

        cbar = None
        if settings.mode == "Continuous":
            cmap = get_continuous_colormap(settings.colormap)
            if combined_mask is not None:
                plot_array = np.ma.array(array, mask=combined_mask)
                try:
                    cmap = cmap.copy()
                except Exception:
                    pass
                cmap.set_bad((0, 0, 0, 0))
            else:
                plot_array = array

            im = ax.imshow(plot_array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if show_colorbar:
                side_text = colorbar_side_override or settings.colorbar_side
                side = "left" if str(side_text).strip().lower() == "left" else "right"
                if colorbar_axis is not None:
                    cax = colorbar_axis
                else:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes(side, size="4%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                set_colorbar_clipping_endpoint_labels(
                    cbar,
                    vmin,
                    vmax,
                    show_lower_as_clipped=actual_min < vmin,
                    show_upper_as_clipped=actual_max > vmax,
                    round_unclipped_lower=not manual_display_min,
                    round_unclipped_upper=not manual_display_max,
                )

        elif settings.mode == "Manual bins":
            boundaries = parse_bins_text(settings.bins)
            colors = parse_color_list(settings.bin_colors)

            custom_cmap = ListedColormap(colors)
            if combined_mask is not None:
                plot_array = np.ma.array(array, mask=combined_mask)
                try:
                    custom_cmap = custom_cmap.copy()
                except Exception:
                    pass
                custom_cmap.set_bad((0, 0, 0, 0))
            else:
                plot_array = array

            norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)
            im = ax.imshow(plot_array, cmap=custom_cmap, norm=norm, interpolation="nearest")
            if show_colorbar:
                side_text = colorbar_side_override or settings.colorbar_side
                side = "left" if str(side_text).strip().lower() == "left" else "right"
                if colorbar_axis is not None:
                    cax = colorbar_axis
                else:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes(side, size="4%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_ticks(boundaries)
                if boundaries:
                    set_colorbar_clipping_endpoint_labels(
                        cbar,
                        boundaries[0],
                        boundaries[-1],
                        show_lower_as_clipped=actual_min < boundaries[0],
                        show_upper_as_clipped=actual_max > boundaries[-1],
                    )

        else:
            raise ValueError(f"Unknown mode '{settings.mode}' for '{output_name}'.")

        for rule_mask, rule_color in (mask_layers or []):
            overlay = np.zeros(rule_mask.shape + (4,), dtype=float)
            overlay[rule_mask] = to_rgba(rule_color)
            ax.imshow(overlay, interpolation="nearest")

        if line_coords is not None:
            start_x, start_y, end_x, end_y = line_coords
            line_length = float(np.hypot(float(end_x) - float(start_x), float(end_y) - float(start_y)))
            if line_length > 0:
                ax.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="red",
                        linewidth=2.0,
                        mutation_scale=16,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=5,
                )
            ax.scatter([start_x], [start_y], color="red", s=24, zorder=6)
            if show_line_length_label:
                self.draw_line_length_label(ax, line_coords, pixel_size_um, array.shape)

        if add_scale_bar:
            self.draw_scale_bar(ax, array.shape, pixel_size_um)

        if cbar is not None:
            guide_label = (colorbar_label_override or "").strip() or settings.colorbar_label.strip() or output_name
            cbar.set_label(guide_label, fontname=DEFAULT_FONT_NAME)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontname(DEFAULT_FONT_NAME)
            side_text = colorbar_side_override or settings.colorbar_side
            if str(side_text).strip().lower() == "left":
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.tick_left()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def save_line_profile_figure(self, profile_data: dict, save_path: str, dpi: int = 300) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        self.plot_line_profile_axis(ax, profile_data)
        if normalize_line_profile_legend_position(self.state.line_profile_settings.legend_position) == "Outside right":
            fig.subplots_adjust(right=0.72)
        else:
            fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    def save_line_profile_combined_figure(
        self,
        profile_data: dict,
        output_arrays: Dict[str, np.ndarray],
        raw_arrays: Dict[str, np.ndarray],
        save_path: str,
        dpi: int = 300,
        add_scale_bar: bool = False,
        pixel_size_um: Optional[float] = None,
    ) -> None:
        base_output_name = profile_data["base_output_name"]
        if base_output_name not in output_arrays:
            raise ValueError(f"Base output '{base_output_name}' is not available for combined line-profile export.")
        array = output_arrays[base_output_name]
        display_settings = self.state.display_settings.get(
            base_output_name,
            OutputDisplaySettings(
                mode="Continuous",
                colormap="gray" if is_grayscale_name(base_output_name) else "viridis",
                colorbar_label=default_colorbar_label(base_output_name),
            ),
        )
        mask_layers = self.build_grayscale_mask_layers(raw_arrays, output_name=base_output_name)
        line_coords = (
            profile_data["start_x"],
            profile_data["start_y"],
            profile_data["end_x"],
            profile_data["end_y"],
        )

        fig = plt.figure(figsize=(11, 4.8), dpi=dpi)

        # Keep the map panel and line-profile panel the same physical size.
        # Colorbar/labels live outside the map panel and therefore do not count
        # toward the main panel width/height when the figure is used side by side.
        panel_y = 0.16
        panel_h = 0.68
        panel_w = 0.32
        map_x = 0.12
        profile_x = 0.53
        ax_map = fig.add_axes([map_x, panel_y, panel_w, panel_h])
        ax_profile = fig.add_axes([profile_x, panel_y, panel_w, panel_h])
        cax = fig.add_axes([map_x - 0.026, panel_y, 0.014, panel_h])

        self.plot_output_map_axis(
            ax_map,
            array,
            base_output_name,
            display_settings,
            mask_layers=mask_layers,
            line_coords=line_coords,
            show_colorbar=True,
            add_scale_bar=add_scale_bar,
            pixel_size_um=pixel_size_um,
            colorbar_axis=cax,
            colorbar_side_override="left",
            colorbar_label_override=self.state.line_profile_settings.image_y_label.strip() or None,
            show_line_length_label=True,
        )
        lp_settings = self.state.line_profile_settings
        if lp_settings.image_title.strip():
            ax_map.set_title(lp_settings.image_title.strip(), fontname=DEFAULT_FONT_NAME)
        if lp_settings.image_x_label.strip():
            ax_map.set_xlabel(lp_settings.image_x_label.strip(), fontname=DEFAULT_FONT_NAME)

        self.plot_line_profile_axis(
            ax_profile,
            profile_data,
            title_override=self.state.line_profile_settings.title.strip(),
        )

        # If the image aspect ratio causes Matplotlib to shrink the displayed map box,
        # match the profile axes to the final map axes size. This keeps the two main
        # panels visually equal; labels, legends, and colorbars may extend outside.
        try:
            fig.canvas.draw()
            map_pos = ax_map.get_position()
            profile_pos = ax_profile.get_position()
            cax.set_position([map_pos.x0 - 0.022, map_pos.y0, 0.012, map_pos.height])
            new_profile_position = [profile_pos.x0, map_pos.y0, map_pos.width, map_pos.height]
            ax_profile.set_position(new_profile_position)
            right_profile_ax = getattr(ax_profile, "_elementti_right_axis", None)
            if right_profile_ax is not None:
                right_profile_ax.set_position(new_profile_position)
        except Exception:
            pass

        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    def compute_output_stats(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        raw_arrays = self.load_raw_arrays()
        working_arrays = self.apply_processing_rules(raw_arrays)

        result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for name, arr in working_arrays.items():
            display_mask = self.build_combined_grayscale_mask(raw_arrays, output_name=name)
            finite = self.get_visible_finite_values(arr, mask=display_mask)
            result[name] = (
                None if finite.size == 0 else float(np.min(finite)),
                None if finite.size == 0 else float(np.max(finite)),
            )

        for num, den in self.state.selected_ratios:
            if num not in working_arrays or den not in working_arrays:
                continue

            ratio_arr = self.calculate_ratio_array(working_arrays, num, den)

            ratio_name = f"{num} / {den}"
            display_mask = self.build_combined_grayscale_mask(raw_arrays, output_name=ratio_name)
            finite = self.get_visible_finite_values(ratio_arr, mask=display_mask)
            result[ratio_name] = (
                None if finite.size == 0 else float(np.min(finite)),
                None if finite.size == 0 else float(np.max(finite)),
            )

        for formula in self.state.selected_formulas:
            formula_name = formula.name.strip()
            if formula_name == "":
                continue
            formula_arr = self.calculate_formula_array(working_arrays, formula_name, formula.expression)
            display_mask = self.build_combined_grayscale_mask(raw_arrays, output_name=formula_name)
            finite = self.get_visible_finite_values(formula_arr, mask=display_mask)
            result[formula_name] = (
                None if finite.size == 0 else float(np.min(finite)),
                None if finite.size == 0 else float(np.max(finite)),
            )

        self.state.processed_min_max_by_name = result
        return result

    def save_map_figure(
        self,
        array: np.ndarray,
        output_name: str,
        settings: OutputDisplaySettings,
        save_path: str,
        dpi: int = 300,
        mask_layers: Optional[List[Tuple[np.ndarray, str]]] = None,
        add_scale_bar: bool = False,
        pixel_size_um: Optional[float] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        self.plot_output_map_axis(
            ax,
            array,
            output_name,
            settings,
            mask_layers=mask_layers,
            line_coords=None,
            show_colorbar=True,
            add_scale_bar=add_scale_bar,
            pixel_size_um=pixel_size_um,
        )
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    def build_summary_dict(self) -> dict:
        renamed_files = {
            self.state.renamed_names_by_path.get(path, ""): path
            for path in self.state.selected_file_paths
        }

        processing = {}
        for path in self.state.selected_file_paths:
            name = self.state.renamed_names_by_path.get(path, "")
            settings = self.state.file_processing_settings.get(path, FileReadSettings())
            processing[name] = {
                "manual_rows": settings.manual_rows,
                "manual_columns": settings.manual_columns,
                "scale_factor": settings.scale_factor,
                "pixel_size_um_per_pixel": settings.pixel_size_um,
            }

        low_value_by_name = {}
        for name, rule in self.state.masking_and_noise_settings.low_value_by_name.items():
            low_value_by_name[name] = {
                "enabled": rule.enabled,
                "floor": rule.floor,
            }

        grayscale_rules = []
        for rule in self.state.masking_and_noise_settings.grayscale_rules:
            grayscale_rules.append({
                "enabled": rule.enabled,
                "source_name": rule.source_name,
                "target_output_name": rule.target_output_name,
                "condition": normalize_grayscale_condition(rule.condition),
                "value_a": rule.value_a,
                "value_b": rule.value_b,
                "color": rule.color,
            })

        conditional_rules = []
        for rule in self.state.masking_and_noise_settings.conditional_rules:
            conditional_rules.append({
                "enabled": rule.enabled,
                "condition_source_name": rule.condition_source_name,
                "condition_operator": normalize_conditional_operator(rule.condition_operator),
                "value_a": rule.value_a,
                "value_b": rule.value_b,
                "target_name": rule.target_name,
                "replacement_value": rule.replacement_value,
            })

        display_settings = {}
        for name, settings in self.state.display_settings.items():
            display_settings[name] = {
                "mode": settings.mode,
                "colormap": settings.colormap,
                "colorbar_side": settings.colorbar_side,
                "colorbar_label": settings.colorbar_label,
                "display_min": settings.display_min,
                "display_max": settings.display_max,
                "bins": settings.bins,
                "bin_colors": settings.bin_colors,
            }

        lp = self.state.line_profile_settings
        line_profile_settings = {
            "enabled": lp.enabled,
            "base_output_name": lp.base_output_name,
            "selected_output_names": lp.selected_output_names,
            "start_point": {"x": lp.start_x, "y": lp.start_y},
            "end_point": {"x": lp.end_x, "y": lp.end_y},
            "normalize_mode": lp.normalize_mode,
            "image_title": lp.image_title,
            "image_x_label": lp.image_x_label,
            "image_guide_bar_label": lp.image_y_label,
            "title": lp.title,
            "x_label": lp.x_label,
            "left_y_label": lp.y_label,
            "left_y_min": lp.y_min,
            "left_y_max": lp.y_max,
            "right_y_label": lp.right_y_label,
            "right_y_min": lp.right_y_min,
            "right_y_max": lp.right_y_max,
            "legend_position": normalize_line_profile_legend_position(lp.legend_position),
            "curve_settings": {
                name: {
                    "enabled": settings.enabled,
                    "color": settings.color,
                    "line_width": settings.line_width,
                }
                for name, settings in lp.curve_settings.items()
            },
        }

        return {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "developer": APP_DEVELOPER_TEXT,
            "citation": APP_CITATION_TEXT,
            "sample_data": {
                "used": bool(self.state.sample_data_mode),
                "mode": self.state.sample_data_mode or None,
            },
            "renamed_files": renamed_files,
            "selected_maps": self.state.selected_maps,
            "selected_ratios": [
                {"numerator": num, "denominator": den}
                for num, den in self.state.selected_ratios
            ],
            "selected_formulas": [
                {"name": formula.name, "expression": formula.expression}
                for formula in self.state.selected_formulas
            ],
            "file_processing_settings": processing,
            "masking_and_noise_settings": {
                "enabled": self.state.masking_and_noise_settings.enabled,
                "grayscale_enabled": self.state.masking_and_noise_settings.grayscale_enabled,
                "grayscale_rules": grayscale_rules,
                "low_value_enabled": self.state.masking_and_noise_settings.low_value_enabled,
                "low_value_by_name": low_value_by_name,
                "conditional_enabled": self.state.masking_and_noise_settings.conditional_enabled,
                "conditional_rules": conditional_rules,
            },
            "processed_min_max_by_name": {
                name: {"min": vals[0], "max": vals[1]}
                for name, vals in self.state.processed_min_max_by_name.items()
            },
            "display_settings": display_settings,
            "line_profile_settings": line_profile_settings,
            "scale_bar_settings": {
                "enabled": self.state.scale_bar_settings.enabled,
                "length_um": self.state.scale_bar_settings.length_um,
                "color": self.state.scale_bar_settings.color,
                "position": self.state.scale_bar_settings.position,
                "pixel_size_source": "per input file in file_processing_settings",
            },
            "output_options": {
                "output_folder": self.state.output_folder,
                "project_name": self.state.project_name,
                "save_figures": self.state.save_png,
                "save_csv": self.state.save_csv,
                "figure_format": self.state.figure_format,
                "figure_dpi": self.state.figure_dpi,
            },
        }

    def build_methods_text(self) -> str:
        lines = []

        lines.append(f"{APP_NAME} methods summary")
        lines.append("")
        lines.append(f"Version: {APP_VERSION}")
        lines.append(APP_DEVELOPER_TEXT)
        lines.append("")
        lines.append("Citation:")
        lines.append(APP_CITATION_TEXT)
        lines.append("")

        lines.append("Demo data:")
        if self.state.sample_data_mode:
            lines.append("- Used")
        else:
            lines.append("- Not used")
        lines.append("")

        lines.append("Input files:")
        for path in self.state.selected_file_paths:
            name = self.state.renamed_names_by_path.get(path, "")
            lines.append(f"- {name}: {path}")
        lines.append("")

        lines.append("Selected maps:")
        if self.state.selected_maps:
            for name in self.state.selected_maps:
                lines.append(f"- {name}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("Selected ratios:")
        if self.state.selected_ratios:
            for num, den in self.state.selected_ratios:
                lines.append(f"- {num} / {den}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("Selected custom formulas:")
        if self.state.selected_formulas:
            for formula in self.state.selected_formulas:
                lines.append(f"- {formula.name} = {formula.expression}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("File processing settings:")
        for path in self.state.selected_file_paths:
            name = self.state.renamed_names_by_path.get(path, "")
            settings = self.state.file_processing_settings.get(path, FileReadSettings())
            lines.append(f"- {name}")
            lines.append(f"  Rows removed: {settings.manual_rows if settings.manual_rows else 'None'}")
            lines.append(f"  Columns removed: {settings.manual_columns if settings.manual_columns else 'None'}")
            lines.append(f"  Scale factor: {settings.scale_factor}")
            lines.append(f"  Pixel size (µm/pixel): {settings.pixel_size_um if settings.pixel_size_um else 'None'}")
        lines.append("")

        ms = self.state.masking_and_noise_settings
        lines.append("Masking and noise-removal settings:")
        lines.append(f"- Enabled: {'Yes' if ms.enabled else 'No'}")

        lines.append(f"- Grayscale masking enabled: {'Yes' if ms.grayscale_enabled else 'No'}")
        if ms.grayscale_enabled:
            active_grayscale_rules = [rule for rule in ms.grayscale_rules if rule.enabled]
            if active_grayscale_rules:
                for idx, rule in enumerate(active_grayscale_rules, start=1):
                    condition = normalize_grayscale_condition(rule.condition)
                    if condition == "between":
                        condition_text = f"{rule.source_name} between {rule.value_a} and {rule.value_b}"
                    elif condition == "outside":
                        condition_text = f"{rule.source_name} outside {rule.value_a} and {rule.value_b}"
                    elif condition == "above":
                        condition_text = f"{rule.source_name} above {rule.value_a}"
                    else:
                        condition_text = f"{rule.source_name} below {rule.value_a}"

                    lines.append(
                        f"  - Rule {idx}: apply to {rule.target_output_name}; mask pixels where {condition_text}"
                    )
                    lines.append(f"    Color: {rule.color}")
            else:
                lines.append("  - None")

        lines.append(f"- Low-value protection enabled: {'Yes' if ms.low_value_enabled else 'No'}")
        if ms.low_value_enabled:
            any_low_rules = False
            for name, rule in ms.low_value_by_name.items():
                if rule.enabled:
                    lines.append(f"  - {name}: minimum allowed value = {rule.floor}")
                    any_low_rules = True
            if not any_low_rules:
                lines.append("  - None")

        lines.append(f"- Conditional replacement enabled: {'Yes' if ms.conditional_enabled else 'No'}")
        if ms.conditional_enabled:
            active_conditional_rules = [rule for rule in ms.conditional_rules if rule.enabled]
            if active_conditional_rules:
                for idx, rule in enumerate(active_conditional_rules, start=1):
                    operator = normalize_conditional_operator(rule.condition_operator)
                    if operator == "between":
                        condition_text = f"{rule.condition_source_name} between {rule.value_a} and {rule.value_b}"
                    elif operator == "above":
                        condition_text = f"{rule.condition_source_name} above {rule.value_a}"
                    else:
                        condition_text = f"{rule.condition_source_name} below {rule.value_a}"
                    lines.append(
                        f"  - Rule {idx}: if {condition_text}, set {rule.target_name} = {rule.replacement_value}"
                    )
            else:
                lines.append("  - None")
        lines.append("")

        lines.append("Display settings:")
        if self.state.display_settings:
            for output_name, settings in self.state.display_settings.items():
                lines.append(f"- {output_name}")
                lines.append(f"  Mode: {settings.mode}")
                lines.append(f"  Colormap: {settings.colormap}")
                lines.append(f"  Guide bar side: {settings.colorbar_side}")
                lines.append(f"  Guide bar label: {settings.colorbar_label}")
                lines.append(f"  Display min: {settings.display_min if settings.display_min else 'Processed min'}")
                lines.append(f"  Display max: {settings.display_max if settings.display_max else 'Processed max'}")
                if settings.mode == "Manual bins":
                    lines.append(f"  Bins: {settings.bins}")
                    lines.append(f"  Bin colors: {settings.bin_colors}")
        else:
            lines.append("- None")
        lines.append("")

        sb = self.state.scale_bar_settings
        lines.append("Scale-bar settings:")
        lines.append(f"- Enabled: {'Yes' if sb.enabled else 'No'}")
        lines.append("- Pixel size source: per input file")
        lines.append(f"- Scale-bar length (µm): {sb.length_um if sb.length_um else 'Automatic'}")
        lines.append(f"- Color: {sb.color}")
        lines.append(f"- Position: {sb.position}")
        lines.append("")

        lp = self.state.line_profile_settings
        lines.append("Line profile settings:")
        lines.append(f"- Enabled: {'Yes' if lp.enabled else 'No'}")
        if lp.enabled:
            lines.append(f"- Base output: {lp.base_output_name}")
            lines.append(f"- Profiled outputs: {', '.join(lp.selected_output_names) if lp.selected_output_names else 'None'}")
            lines.append(f"- Start point: x={lp.start_x}, y={lp.start_y}")
            lines.append(f"- End point: x={lp.end_x}, y={lp.end_y}")
            lines.append(f"- Normalization: {lp.normalize_mode}")
            lines.append(f"- Image title: {lp.image_title if lp.image_title else 'None'}")
            lines.append(f"- Image X-axis label: {lp.image_x_label if lp.image_x_label else 'None'}")
            lines.append(f"- Image guide-bar label: {lp.image_y_label if lp.image_y_label else 'None'}")
            lines.append(f"- Line-profile title: {lp.title if lp.title else 'None'}")
            lines.append(f"- Line-profile X-axis label: {lp.x_label if lp.x_label else 'Automatic'}")
            lines.append(f"- Line-profile left Y-axis label: {lp.y_label if lp.y_label else 'Automatic'}")
            lines.append(f"- Line-profile left Y-axis range: {lp.y_min if lp.y_min else 'Automatic'} to {lp.y_max if lp.y_max else 'Automatic'}")
            lines.append(f"- Line-profile right Y-axis label: {lp.right_y_label if lp.right_y_label else 'Ratio value'}")
            lines.append(f"- Line-profile right Y-axis range: {lp.right_y_min if lp.right_y_min else 'Automatic'} to {lp.right_y_max if lp.right_y_max else 'Automatic'}")
            lines.append(f"- Line-profile legend position: {normalize_line_profile_legend_position(lp.legend_position)}")
            if lp.curve_settings:
                lines.append("- Curve settings:")
                for output_name in lp.selected_output_names:
                    curve = lp.curve_settings.get(output_name, LineProfileCurveSettings())
                    lines.append(
                        f"  - {output_name}: color={curve.color if curve.color else 'automatic'}, "
                        f"line width={curve.line_width if curve.line_width else '2.0'}"
                    )
        lines.append("")

        lines.append("Output options:")
        lines.append(f"- Output folder: {self.state.output_folder}")
        lines.append(f"- Project name: {self.state.project_name if self.state.project_name else 'None'}")
        lines.append(f"- Save figure files: {'Yes' if self.state.save_png else 'No'}")
        lines.append(f"- Save processed CSV files: {'Yes' if self.state.save_csv else 'No'}")
        lines.append(f"- Figure format: {self.state.figure_format}")
        lines.append(f"- Figure DPI: {self.state.figure_dpi}")
        lines.append(f"- Additional figures with scale bar: {'Yes' if self.state.scale_bar_settings.enabled else 'No'}")

        return "\n".join(lines)

    def generate_outputs(self) -> Tuple[str, str]:
        output_folder = self.state.output_folder.strip()
        if output_folder == "":
            raise ValueError("Output folder is empty.")

        os.makedirs(output_folder, exist_ok=True)

        prefix = sanitize_filename(self.state.project_name.strip()) if self.state.project_name.strip() else ""
        ext = self.state.figure_format.strip().lower()
        dpi = int(self.state.figure_dpi)
        reserved_output_paths = set()

        def output_path(file_stub: str, extension: str) -> str:
            return make_unique_output_path(output_folder, file_stub, extension, reserved_output_paths)

        def add_prefix(file_stub: str) -> str:
            safe_stub = sanitize_filename(file_stub)
            return f"{prefix}_{safe_stub}" if prefix else safe_stub

        raw_arrays = self.load_raw_arrays()
        working_arrays = self.apply_processing_rules(raw_arrays)

        for map_name in self.state.selected_maps:
            if map_name not in working_arrays:
                raise ValueError(f"Selected map '{map_name}' was not loaded.")

            arr = working_arrays[map_name]
            settings = self.state.display_settings[map_name]
            mask_layers = self.build_grayscale_mask_layers(raw_arrays, output_name=map_name)
            file_stub = add_prefix(map_name)

            if self.state.save_csv:
                csv_path = output_path(file_stub, "csv")
                np.savetxt(csv_path, arr, delimiter=",", fmt="%.9g")

            if self.state.save_png:
                fig_path = output_path(file_stub, ext)
                self.save_map_figure(
                    arr,
                    map_name,
                    settings,
                    fig_path,
                    dpi=dpi,
                    mask_layers=mask_layers,
                )

                if self.state.scale_bar_settings.enabled:
                    pixel_size_um = self.get_pixel_size_um_for_output(map_name)
                    if pixel_size_um is None:
                        raise ValueError(f"Scale-bar export is enabled, but no pixel size is available for '{map_name}'.")
                    scale_fig_path = output_path(f"{file_stub}_with_scale", ext)
                    self.save_map_figure(
                        arr,
                        map_name,
                        settings,
                        scale_fig_path,
                        dpi=dpi,
                        mask_layers=mask_layers,
                        add_scale_bar=True,
                        pixel_size_um=pixel_size_um,
                    )

        for numerator, denominator in self.state.selected_ratios:
            if numerator not in working_arrays:
                raise ValueError(f"Ratio numerator '{numerator}' was not loaded.")
            if denominator not in working_arrays:
                raise ValueError(f"Ratio denominator '{denominator}' was not loaded.")

            num_arr = working_arrays[numerator]
            den_arr = working_arrays[denominator]

            if num_arr.shape != den_arr.shape:
                raise ValueError(
                    f"Ratio '{numerator} / {denominator}' cannot be calculated because the two arrays "
                    f"have different shapes: {num_arr.shape} and {den_arr.shape}."
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_arr = num_arr / den_arr
                ratio_arr[~np.isfinite(ratio_arr)] = np.nan

            ratio_name = f"{numerator} / {denominator}"
            settings = self.state.display_settings[ratio_name]
            mask_layers = self.build_grayscale_mask_layers(raw_arrays, output_name=ratio_name)
            file_stub = add_prefix(f"{numerator}_over_{denominator}")

            if self.state.save_csv:
                csv_path = output_path(file_stub, "csv")
                np.savetxt(csv_path, ratio_arr, delimiter=",", fmt="%.9g")

            if self.state.save_png:
                fig_path = output_path(file_stub, ext)
                self.save_map_figure(
                    ratio_arr,
                    ratio_name,
                    settings,
                    fig_path,
                    dpi=dpi,
                    mask_layers=mask_layers,
                )

                if self.state.scale_bar_settings.enabled:
                    pixel_size_um = self.get_pixel_size_um_for_output(ratio_name)
                    if pixel_size_um is None:
                        raise ValueError(f"Scale-bar export is enabled, but no pixel size is available for '{ratio_name}'.")
                    scale_fig_path = output_path(f"{file_stub}_with_scale", ext)
                    self.save_map_figure(
                        ratio_arr,
                        ratio_name,
                        settings,
                        scale_fig_path,
                        dpi=dpi,
                        mask_layers=mask_layers,
                        add_scale_bar=True,
                        pixel_size_um=pixel_size_um,
                    )

        for formula in self.state.selected_formulas:
            formula_name = formula.name.strip()
            if formula_name == "":
                continue

            formula_arr = self.calculate_formula_array(working_arrays, formula_name, formula.expression)
            settings = self.state.display_settings[formula_name]
            mask_layers = self.build_grayscale_mask_layers(raw_arrays, output_name=formula_name)
            file_stub = add_prefix(formula_name)

            if self.state.save_csv:
                csv_path = output_path(file_stub, "csv")
                np.savetxt(csv_path, formula_arr, delimiter=",", fmt="%.9g")

            if self.state.save_png:
                fig_path = output_path(file_stub, ext)
                self.save_map_figure(
                    formula_arr,
                    formula_name,
                    settings,
                    fig_path,
                    dpi=dpi,
                    mask_layers=mask_layers,
                )

                if self.state.scale_bar_settings.enabled:
                    pixel_size_um = self.get_pixel_size_um_for_output(formula_name)
                    if pixel_size_um is None:
                        raise ValueError(f"Scale-bar export is enabled, but no pixel size is available for '{formula_name}'.")
                    scale_fig_path = output_path(f"{file_stub}_with_scale", ext)
                    self.save_map_figure(
                        formula_arr,
                        formula_name,
                        settings,
                        scale_fig_path,
                        dpi=dpi,
                        mask_layers=mask_layers,
                        add_scale_bar=True,
                        pixel_size_um=pixel_size_um,
                    )

        lp = self.state.line_profile_settings
        if lp.enabled:
            if lp.start_x is None or lp.start_y is None or lp.end_x is None or lp.end_y is None:
                raise ValueError("Line-profile export is enabled, but no complete line has been drawn.")
            if not lp.base_output_name:
                raise ValueError("Line-profile export is enabled, but no base output is selected.")
            if not lp.selected_output_names:
                raise ValueError("Line-profile export is enabled, but no profile outputs are selected.")

            output_arrays = {name: np.array(arr, copy=True) for name, arr in working_arrays.items()}
            output_arrays.update(self.build_output_arrays(raw_arrays, working_arrays))
            profile_data = self.extract_line_profile_data(
                output_arrays,
                lp.base_output_name,
                lp.selected_output_names,
                lp.start_x,
                lp.start_y,
                lp.end_x,
                lp.end_y,
                lp.normalize_mode,
            )

            profile_stub = add_prefix("line_profile")

            if self.state.save_csv:
                profile_csv_path = output_path(profile_stub, "csv")
                self.save_line_profile_csv(profile_data, profile_csv_path)

            if self.state.save_png:
                profile_fig_path = output_path(profile_stub, ext)
                self.save_line_profile_figure(profile_data, profile_fig_path, dpi=dpi)

                combined_stub = add_prefix(f"{lp.base_output_name}_with_line_profile")
                combined_fig_path = output_path(combined_stub, ext)
                line_pixel_size_um = None
                try:
                    line_pixel_size_um = self.get_pixel_size_um_for_output(lp.base_output_name)
                except Exception:
                    line_pixel_size_um = None
                self.save_line_profile_combined_figure(
                    profile_data,
                    output_arrays,
                    raw_arrays,
                    combined_fig_path,
                    dpi=dpi,
                    pixel_size_um=line_pixel_size_um,
                )

                if self.state.scale_bar_settings.enabled:
                    pixel_size_um = self.get_pixel_size_um_for_output(lp.base_output_name)
                    if pixel_size_um is None:
                        raise ValueError(f"Scale-bar export is enabled, but no pixel size is available for line-profile base output '{lp.base_output_name}'.")
                    combined_scale_stub = add_prefix(f"{lp.base_output_name}_with_line_profile_and_scale")
                    combined_scale_fig_path = output_path(combined_scale_stub, ext)
                    self.save_line_profile_combined_figure(
                        profile_data,
                        output_arrays,
                        raw_arrays,
                        combined_scale_fig_path,
                        dpi=dpi,
                        add_scale_bar=True,
                        pixel_size_um=pixel_size_um,
                    )

        summary = self.build_summary_dict()
        base_name = prefix if prefix else "elementti_run"
        summary_path = output_path(f"{base_name}_summary", "json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        methods_text = self.build_methods_text()
        methods_path = output_path("methods_Elementti", "txt")

        with open(methods_path, "w", encoding="utf-8") as f:
            f.write(methods_text)

        return summary_path, methods_path


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle(f"Welcome to {APP_NAME}")
        self.setSubTitle(f"Version {APP_VERSION}")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)

        icon_path = resource_path("pngElementti_1_0_8.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.logo_label.setPixmap(
                pixmap.scaled(
                    420,
                    420,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        layout.addWidget(self.logo_label)

        info = QLabel("")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        layout.addStretch()
        set_page_layout(self, layout)


class UploadPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 1: Upload your CSV files")
        self.setSubTitle("Add input files.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Add CSV files or load built-in demo data.",
            "Upload help",
            "Use this page to add the CSV files that contain your exported SEM-EDS map data. Each CSV file should normally correspond to one elemental map or one reference image, such as Cl, Fe, O, or greyscale.\n\n"
            "Click 'Choose CSV files' to select one or more files from your computer. You can return to the button and add more files if needed. Duplicate file paths are ignored.\n\n"
            "The demo button creates built-in example CSV files so that you can test the workflow without your own SEM-EDS data. The demo is fixed, so it loads the same example maps each time. The demo CSV files include five metadata rows and a detected pixel size, matching the default preprocessing settings.\n\n"
            "If you remove or clear files, Elementti resets later selections such as names, ratios, masking, display settings, and line-profile settings, because those settings depend on the selected input files."
        )

        row = QHBoxLayout()
        self.choose_button = QPushButton("Choose CSV files")
        self.demo_sample_button = QPushButton("Load demo data")
        self.remove_button = QPushButton("Remove selected file")
        self.clear_button = QPushButton("Clear all")

        self.choose_button.clicked.connect(self.choose_files)
        self.demo_sample_button.clicked.connect(self.load_demo_sample_data)
        self.remove_button.clicked.connect(self.remove_selected_file)
        self.clear_button.clicked.connect(self.clear_all_files)

        row.addWidget(self.choose_button)
        row.addWidget(self.demo_sample_button)
        row.addWidget(self.remove_button)
        row.addWidget(self.clear_button)
        row.addStretch()
        layout.addLayout(row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.file_list)

        self.status_label = QLabel("No files uploaded yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.refresh_list()

    def isComplete(self):
        return len(self.state.selected_file_paths) > 0

    def refresh_list(self):
        self.file_list.clear()
        for path in self.state.selected_file_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.file_list.addItem(item)

    def choose_files(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose one or more CSV files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_names:
            return

        added = 0
        for path in file_names:
            if path not in self.state.selected_file_paths:
                self.state.selected_file_paths.append(path)
                added += 1

        if added > 0:
            self.state.sample_data_mode = ""
            self.state.sample_data_seed = None
            self.state.reset_all_dependent_settings()

        self.refresh_list()
        self.status_label.setText(
            f"{added} new file(s) added. Click Next when you are done."
            if added > 0
            else "Those files were already added."
        )
        self.completeChanged.emit()

    def load_demo_sample_data(self):
        self._load_sample_data()

    def _load_sample_data(self):
        try:
            # Use a fresh temporary folder for each demo load. This avoids
            # overwriting files that may still be referenced by previews or by
            # the operating system, which can be fragile on Windows.
            old_sample_folder = self.state.sample_data_folder
            sample_folder = tempfile.mkdtemp(prefix="elementti_demo_")
            sample_paths, used_seed = create_sample_data_files(sample_folder, seed=DEFAULT_SAMPLE_SEED)

            self.state.selected_file_paths = list(sample_paths)
            self.state.renamed_names_by_path = {}
            self.state.file_processing_settings = {}
            self.state.sample_data_folder = sample_folder
            self.state.sample_data_mode = "demo"
            self.state.sample_data_seed = used_seed
            self.state.reset_all_dependent_settings()
            if old_sample_folder and os.path.abspath(old_sample_folder) != os.path.abspath(sample_folder):
                cleanup_demo_sample_folder(old_sample_folder)

            self.refresh_list()
            self.status_label.setText(
                f"{len(sample_paths)} demo file(s) loaded. The demo data include "
                f"grayscale_demo.csv, element1_demo.csv, and element2_demo.csv "
                f"with five metadata rows, a detected pixel size, and 384 × 512 data pixels."
            )
            self.completeChanged.emit()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Could not create demo data",
                f"The app could not create the demo CSV files.\n\n{str(e)}",
            )

    def remove_selected_file(self):
        row = self.file_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No file selected", "Please select a file first.")
            return

        item = self.file_list.item(row)
        path = item.data(Qt.UserRole)

        if path in self.state.selected_file_paths:
            self.state.selected_file_paths.remove(path)

        self.state.renamed_names_by_path.pop(path, None)
        self.state.file_processing_settings.pop(path, None)
        self.state.reset_all_dependent_settings()

        if not self.state.selected_file_paths:
            cleanup_demo_sample_folder(self.state.sample_data_folder)
            self.state.sample_data_folder = ""
            self.state.sample_data_mode = ""
            self.state.sample_data_seed = None
        else:
            cleanup_unused_demo_sample_folder(self.state)

        self.refresh_list()
        self.status_label.setText("One file was removed.")
        self.completeChanged.emit()

    def clear_all_files(self):
        if not self.state.selected_file_paths:
            return

        reply = QMessageBox.question(
            self,
            "Clear all files",
            "Remove all uploaded files?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            old_sample_folder = self.state.sample_data_folder
            self.state.selected_file_paths.clear()
            self.state.renamed_names_by_path.clear()
            self.state.file_processing_settings.clear()
            self.state.sample_data_folder = ""
            self.state.sample_data_mode = ""
            self.state.sample_data_seed = None
            self.state.reset_all_dependent_settings()
            cleanup_demo_sample_folder(old_sample_folder)
            self.refresh_list()
            self.status_label.setText("All uploaded files were cleared.")
            self.completeChanged.emit()


class RenamePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 2: Rename your files")
        self.setSubTitle("Assign short names.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Give each imported file a short unique name used later in the workflow.",
            "Rename help",
            "Use this page to give each imported CSV file a short, meaningful name. These names are used throughout the rest of Elementti: for selecting maps, defining ratios, setting masks, choosing display settings, creating line profiles, and naming exported files.\n\n"
            "The left column shows the original file name. The right column contains the name that Elementti will use internally. Examples are Cl, Fe, O, Ca, P, greyscale, Cl500X, or Fe100X.\n\n"
            "Each name must be unique. Avoid using the same name for two different files. If you change names after later settings have already been selected, Elementti resets dependent settings so that ratios and masks do not refer to old names."
        )

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Original CSV file", "Name to use in the app"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget::item:selected {
                background: white;
                color: black;
            }
        """)
        self.table.cellClicked.connect(self.start_editing_name_cell)
        layout.addWidget(self.table)

        self.status_label = QLabel("Click a cell in the right column to rename it.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.populate_table()

    def populate_table(self):
        self.table.setRowCount(0)
        used_names = set()

        for path in self.state.selected_file_paths:
            row = self.table.rowCount()
            self.table.insertRow(row)

            original_name = os.path.basename(path)
            original_item = QTableWidgetItem(original_name)
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
            original_item.setData(Qt.UserRole, path)

            chosen_name = self.state.renamed_names_by_path.get(path)
            if chosen_name is None:
                chosen_name = suggest_name(original_name, used_names)

            used_names.add(chosen_name)

            self.table.setItem(row, 0, original_item)
            self.table.setItem(row, 1, QTableWidgetItem(chosen_name))

    def start_editing_name_cell(self, row, column):
        if column != 1:
            return
        item = self.table.item(row, column)
        if item is None:
            return
        self.table.editItem(item)
        QTimer.singleShot(0, self.select_all_editor_text)

    def select_all_editor_text(self):
        editor = QApplication.focusWidget()
        if editor is not None and hasattr(editor, "selectAll"):
            editor.selectAll()

    def validatePage(self):
        names = []
        result = {}

        for row in range(self.table.rowCount()):
            original_item = self.table.item(row, 0)
            chosen_item = self.table.item(row, 1)

            if original_item is None or chosen_item is None:
                QMessageBox.warning(self, "Incomplete row", "One row is incomplete.")
                return False

            original_name = original_item.text()
            path = original_item.data(Qt.UserRole)
            chosen_name = chosen_item.text().strip()

            if not chosen_name:
                QMessageBox.warning(self, "Empty name", f"Please enter a name for '{original_name}'.")
                return False

            names.append(chosen_name)
            result[path] = chosen_name

        old_names_by_path = dict(self.state.renamed_names_by_path)

        if len(names) != len(set(names)):
            QMessageBox.warning(self, "Duplicate names", "Each chosen name must be unique.")
            return False

        self.state.renamed_names_by_path = result

        if old_names_by_path != result:
            self.state.reset_all_dependent_settings()

        self.status_label.setText("Renaming finished successfully.")
        return True



class OutputsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 3: Choose what you want to generate")
        self.setSubTitle("Select maps, ratios, and custom formulas.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Select direct maps, ratio maps, and optional custom formula maps to generate.",
            "Output selection help",
            "Use this page to decide which outputs Elementti should generate. Direct maps are the individual imported arrays, such as Cl, Fe, or greyscale. Ratio maps are calculated from two selected arrays, for example Cl / Fe or Ca / P.\n\n"
            "Tick a file name if you want Elementti to generate it as a direct map. To create a ratio map, choose the numerator and denominator from the drop-down boxes and click 'Add ratio'.\n\n"
            "Custom formulas let you create calculated maps such as X + Y / Z, (X + Y) / Z, X / (Y + Z), sqrt(X), or log10(X). Use input-map names directly when they are simple names like Cl or Fe. For names with spaces or symbols, write the exact input name in square brackets or braces, for example [My map] / Fe or {My map} / Fe. Supported operators are +, -, *, /, **, parentheses, unary +/- and the functions sqrt(), log(), ln(), log10(), log2(), exp(), abs(), sin(), cos(), tan(), min(), max(), minimum(), maximum(), and clip().\n\n"
            "Choose ratios and formulas carefully: very small denominator values can create unstable or very large values. Later pages allow low-value protection, conditional replacement, and display masking to help manage these cases. Non-finite results, such as division by zero or invalid logarithms, are stored as NaN."
        )

        layout.addWidget(QLabel("Maps to plot:"))
        self.maps_list = QListWidget()
        self.maps_list.itemChanged.connect(self.on_maps_changed)
        layout.addWidget(self.maps_list)

        layout.addWidget(QLabel("Ratios to create:"))

        ratio_row = QHBoxLayout()
        self.numerator_combo = QComboBox()
        self.denominator_combo = QComboBox()
        self.add_ratio_button = QPushButton("Add ratio")
        self.add_ratio_button.clicked.connect(self.add_ratio)

        ratio_row.addWidget(QLabel("Numerator"))
        ratio_row.addWidget(self.numerator_combo)
        ratio_row.addWidget(QLabel("Denominator"))
        ratio_row.addWidget(self.denominator_combo)
        ratio_row.addWidget(self.add_ratio_button)
        layout.addLayout(ratio_row)

        self.ratio_list = QListWidget()
        self.ratio_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.ratio_list)

        ratio_remove_row = QHBoxLayout()
        self.remove_ratio_button = QPushButton("Remove selected ratio")
        self.remove_ratio_button.clicked.connect(self.remove_selected_ratio)
        ratio_remove_row.addWidget(self.remove_ratio_button)
        ratio_remove_row.addStretch()
        layout.addLayout(ratio_remove_row)

        layout.addWidget(QLabel("Custom formulas:"))

        formula_row = QHBoxLayout()
        self.formula_name_edit = QLineEdit()
        self.formula_name_edit.setPlaceholderText("Output name, e.g. Cl_plus_Fe_over_O")
        self.formula_expression_edit = QLineEdit()
        self.formula_expression_edit.setPlaceholderText("Formula, e.g. (Cl + Fe) / O")
        self.add_formula_button = QPushButton("Add formula")
        self.add_formula_button.clicked.connect(self.add_formula)
        formula_row.addWidget(QLabel("Name"))
        formula_row.addWidget(self.formula_name_edit, stretch=1)
        formula_row.addWidget(QLabel("Expression"))
        formula_row.addWidget(self.formula_expression_edit, stretch=2)
        formula_row.addWidget(self.add_formula_button)
        layout.addLayout(formula_row)

        self.formula_list = QListWidget()
        self.formula_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.formula_list)

        formula_remove_row = QHBoxLayout()
        self.remove_formula_button = QPushButton("Remove selected formula")
        self.remove_formula_button.clicked.connect(self.remove_selected_formula)
        formula_remove_row.addWidget(self.remove_formula_button)
        formula_remove_row.addStretch()
        layout.addLayout(formula_remove_row)

        self.status_label = QLabel("Choose at least one map, ratio, or formula.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.populate_maps_list()
        self.populate_ratio_combos()
        self.populate_ratio_list()
        self.populate_formula_list()
        self.update_status()

    def isComplete(self):
        return (
            len(self.get_selected_maps()) > 0
            or len(self.state.selected_ratios) > 0
            or len(self.state.selected_formulas) > 0
        )

    def populate_maps_list(self):
        self.maps_list.blockSignals(True)
        self.maps_list.clear()
        for name in self.state.get_ordered_names():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in self.state.selected_maps else Qt.Unchecked)
            self.maps_list.addItem(item)
        self.maps_list.blockSignals(False)

    def populate_ratio_combos(self):
        names = self.state.get_ordered_names()
        self.numerator_combo.clear()
        self.denominator_combo.clear()
        self.numerator_combo.addItems(names)
        self.denominator_combo.addItems(names)

    def populate_ratio_list(self):
        self.ratio_list.clear()
        for num, den in self.state.selected_ratios:
            self.ratio_list.addItem(f"{num} / {den}")

    def populate_formula_list(self):
        self.formula_list.clear()
        for formula in self.state.selected_formulas:
            self.formula_list.addItem(f"{formula.name} = {formula.expression}")

    def get_selected_maps(self) -> List[str]:
        selected = []
        for i in range(self.maps_list.count()):
            item = self.maps_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def on_maps_changed(self, _item):
        self.state.selected_maps = self.get_selected_maps()
        self.update_status()
        self.completeChanged.emit()

    def current_output_names_without_formula_row(self, formula_row_to_ignore: Optional[int] = None) -> set:
        names = set(self.get_selected_maps())
        names.update(f"{num} / {den}" for num, den in self.state.selected_ratios)
        for idx, formula in enumerate(self.state.selected_formulas):
            if formula_row_to_ignore is not None and idx == formula_row_to_ignore:
                continue
            formula_name = formula.name.strip()
            if formula_name:
                names.add(formula_name)
        return names

    def validate_formula_output_name(self, formula_name: str, formula_row_to_ignore: Optional[int] = None) -> bool:
        if formula_name == "":
            QMessageBox.warning(self, "Missing formula name", "Please enter an output name for the formula.")
            return False
        if formula_name.lower() == "all outputs":
            QMessageBox.warning(self, "Invalid formula name", "'All outputs' is reserved for masking rules. Please choose another formula name.")
            return False
        if any(ch in formula_name for ch in "[]{}=\n\r\t"):
            QMessageBox.warning(self, "Invalid formula name", "Formula output names cannot contain [, ], {, }, =, tabs, or line breaks.")
            return False
        if formula_name in self.state.get_ordered_names():
            QMessageBox.warning(
                self,
                "Formula name conflicts with input map",
                f"'{formula_name}' is already an input-map name. Please choose a different formula output name."
            )
            return False
        if formula_name in self.current_output_names_without_formula_row(formula_row_to_ignore):
            QMessageBox.warning(
                self,
                "Duplicate output name",
                f"'{formula_name}' is already selected as a map, ratio, or formula output. Please choose a unique name."
            )
            return False
        return True

    def add_ratio(self):
        num = self.numerator_combo.currentText().strip()
        den = self.denominator_combo.currentText().strip()

        if not num or not den:
            QMessageBox.warning(self, "Missing selection", "Please choose both numerator and denominator.")
            return

        if num == den:
            QMessageBox.warning(self, "Invalid ratio", "Numerator and denominator cannot be the same.")
            return

        ratio = (num, den)
        ratio_name = f"{num} / {den}"
        if ratio in self.state.selected_ratios:
            QMessageBox.warning(self, "Duplicate ratio", "That ratio has already been added.")
            return
        if any(formula.name.strip() == ratio_name for formula in self.state.selected_formulas):
            QMessageBox.warning(
                self,
                "Duplicate output name",
                f"A custom formula already uses the output name '{ratio_name}'. Rename or remove that formula first."
            )
            return

        self.state.selected_ratios.append(ratio)
        self.populate_ratio_list()
        self.update_status()
        self.completeChanged.emit()

    def remove_selected_ratio(self):
        row = self.ratio_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No ratio selected", "Please select a ratio first.")
            return

        del self.state.selected_ratios[row]
        self.populate_ratio_list()
        self.update_status()
        self.completeChanged.emit()

    def add_formula(self, _checked=False) -> bool:
        formula_name = self.formula_name_edit.text().strip()
        expression = self.formula_expression_edit.text().strip()

        if not self.validate_formula_output_name(formula_name):
            return False
        if expression == "":
            QMessageBox.warning(self, "Missing formula", "Please enter a formula expression.")
            return False

        try:
            compile_formula_expression(expression, self.state.get_ordered_names())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid formula", str(e))
            return False

        self.state.selected_formulas.append(FormulaOutput(name=formula_name, expression=expression))
        self.formula_name_edit.clear()
        self.formula_expression_edit.clear()
        self.populate_formula_list()
        self.update_status()
        self.completeChanged.emit()
        return True

    def remove_selected_formula(self):
        row = self.formula_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No formula selected", "Please select a formula first.")
            return

        del self.state.selected_formulas[row]
        self.populate_formula_list()
        self.update_status()
        self.completeChanged.emit()

    def validatePage(self):
        self.state.selected_maps = self.get_selected_maps()

        if self.formula_name_edit.text().strip() or self.formula_expression_edit.text().strip():
            if not self.add_formula():
                return False

        seen_formula_names = set()
        for idx, formula in enumerate(self.state.selected_formulas):
            formula_name = formula.name.strip()
            expression = formula.expression.strip()
            if formula_name in seen_formula_names:
                QMessageBox.warning(self, "Duplicate formula name", f"Formula name '{formula_name}' is used more than once.")
                return False
            seen_formula_names.add(formula_name)
            if not self.validate_formula_output_name(formula_name, formula_row_to_ignore=idx):
                return False
            try:
                compile_formula_expression(expression, self.state.get_ordered_names())
            except ValueError as e:
                QMessageBox.warning(self, "Invalid formula", f"Formula '{formula_name}' is invalid.\n\n{str(e)}")
                return False

        if not self.state.selected_maps and not self.state.selected_ratios and not self.state.selected_formulas:
            QMessageBox.warning(self, "Nothing selected", "Please choose at least one map, ratio, or formula.")
            return False

        self.state.prune_display_settings_to_current_outputs()
        return True

    def update_status(self):
        self.status_label.setText(
            f"Selected maps: {len(self.get_selected_maps())}    "
            f"Selected ratios: {len(self.state.selected_ratios)}    "
            f"Selected formulas: {len(self.state.selected_formulas)}"
        )


class ProcessingSettingsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 4: Preprocessing and pixel size")
        self.setSubTitle("Set row/column removal, scale factors, and optional pixel sizes.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Remove unwanted rows/columns, set scale factors, and check detected pixel sizes.",
            "Preprocessing help",
            "Use this page to clean the raw CSV table before numerical processing. Many SEM-EDS exports contain header rows, labels, or extra columns that are not part of the numerical map. Those rows or columns must be removed before Elementti can convert the remaining cells into an array.\n\n"
            "Row and column numbers are 1-based, meaning the first row is 1 and the first column is 1. To remove a continuous range, use brackets such as [1-5]. You can also combine ranges and single values, for example [1-5], 17, 28, [94-102], 140, 143. Leave a box blank if no rows or columns should be removed.\n\n"
            "The scale factor is applied after row/column removal. A greyscale image usually uses scale factor 1.0. Elemental maps exported as atomic percent multiplied by 100 often use scale factor 0.01, so that a stored value of 250 becomes 2.5.\n\n"
            "The pixel-size field is optional and is used only if you later choose to export scale-bar figures. Elementti tries to detect pixel size automatically from metadata/header text in each CSV file. If the metadata uses nm, µm/um, mm, cm, or m per pixel, Elementti converts the detected value to µm/pixel before filling the box. For example, 'Pixel Size 1874.568 nm' becomes 1.874568. If Elementti cannot find it, the field stays blank and you can type the value manually if you know it. You can also type values with units, such as 1874.568 nm/pixel or 0.001874568 mm/pixel; they are converted to µm/pixel when you continue.\n\n"
            "All thresholds used later for masking, low-value protection, and conditional replacement are interpreted after the scale-factor step."
        )

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                "Renamed file",
                "Rows to remove",
                "Columns to remove",
                "Scale factor",
                "Pixel size (µm/pixel)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.status_label = QLabel("Set preprocessing values. Pixel size is optional unless you later export scale-bar figures. Pixel-size entries may include nm, µm/um, mm, cm, or m and are stored as µm/pixel.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.populate_table()

    def default_scale_for_name(self, name: str) -> str:
        return "1.0" if is_grayscale_name(name) else "0.01"

    def ensure_defaults(self):
        for path in self.state.selected_file_paths:
            if path not in self.state.file_processing_settings:
                name = self.state.renamed_names_by_path.get(path, "")
                self.state.file_processing_settings[path] = FileReadSettings(
                    manual_rows="[1-5]",
                    manual_columns="",
                    scale_factor=self.default_scale_for_name(name),
                    pixel_size_um=detect_pixel_size_um_from_file(path),
                )

    def populate_table(self):
        self.ensure_defaults()
        self.table.setRowCount(0)

        for path in self.state.selected_file_paths:
            row = self.table.rowCount()
            self.table.insertRow(row)

            renamed = self.state.renamed_names_by_path.get(path, os.path.basename(path))
            settings = self.state.file_processing_settings[path]

            name_item = QTableWidgetItem(renamed)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, path)
            self.table.setItem(row, 0, name_item)

            rows_edit = QLineEdit(settings.manual_rows)
            rows_edit.setPlaceholderText("Example: [1-5], 17, [94-102]")
            self.table.setCellWidget(row, 1, rows_edit)

            cols_edit = QLineEdit(settings.manual_columns)
            cols_edit.setPlaceholderText("Example: [1-3], 8, [12-15]")
            self.table.setCellWidget(row, 2, cols_edit)

            scale_edit = QLineEdit(settings.scale_factor)
            self.table.setCellWidget(row, 3, scale_edit)

            pixel_size_edit = QLineEdit(settings.pixel_size_um)
            pixel_size_edit.setPlaceholderText("optional, e.g. 1.874568 or 1874.568 nm/pixel")
            self.table.setCellWidget(row, 4, pixel_size_edit)

    def validatePage(self):
        result: Dict[str, FileReadSettings] = {}

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            path = name_item.data(Qt.UserRole)
            renamed = name_item.text()

            manual_rows = self.table.cellWidget(row, 1).text().strip()
            manual_columns = self.table.cellWidget(row, 2).text().strip()
            scale_text = self.table.cellWidget(row, 3).text().strip()
            pixel_size_text = self.table.cellWidget(row, 4).text().strip()

            try:
                parse_manual_index_list(manual_rows, "Rows to remove")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid rows", f"For '{renamed}':\n\n{str(e)}")
                return False

            try:
                parse_manual_index_list(manual_columns, "Columns to remove")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid columns", f"For '{renamed}':\n\n{str(e)}")
                return False

            if scale_text == "":
                QMessageBox.warning(self, "Missing scale factor", f"Please enter the scale factor for '{renamed}'.")
                return False

            try:
                scale_value = parse_float_user_text(scale_text, "Scale factor")
                scale_text = format_detected_float(scale_value)
            except ValueError:
                QMessageBox.warning(self, "Invalid scale factor", f"Scale factor for '{renamed}' must be a number.")
                return False

            if pixel_size_text:
                try:
                    pixel_size_value = parse_pixel_size_text_to_um(pixel_size_text, "Pixel size")
                    if pixel_size_value <= 0:
                        raise ValueError
                    pixel_size_text = format_detected_float(pixel_size_value)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid pixel size",
                        f"Pixel size for '{renamed}' must be a number greater than zero, or left blank. "
                        "Accepted units are nm, µm/um, mm, cm, and m per pixel."
                    )
                    return False

            result[path] = FileReadSettings(
                manual_rows=manual_rows,
                manual_columns=manual_columns,
                scale_factor=scale_text,
                pixel_size_um=pixel_size_text,
            )

        old_settings = {
            path: asdict(settings)
            for path, settings in self.state.file_processing_settings.items()
        }
        new_settings = {
            path: asdict(settings)
            for path, settings in result.items()
        }

        self.state.file_processing_settings = result

        if old_settings != new_settings:
            self.state.clear_display_results_only()

        return True


class MaskingIntroPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 5: Masking and noise removal")
        self.setSubTitle("Choose whether to configure optional preprocessing rules.")

        layout = QVBoxLayout()
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("optionalDecisionCard")
        card.setMinimumWidth(760)
        card.setMaximumWidth(1100)
        card.setMinimumHeight(220)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.setStyleSheet(
            "QFrame#optionalDecisionCard { background: white; border: 1px solid #d6d6d6; border-radius: 8px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.setSpacing(14)

        title_label = QLabel("Optional masking and noise removal")
        title_label.setStyleSheet("font-weight: 700; font-size: 15px;")
        card_layout.addWidget(title_label)

        summary_label = QLabel(
            "Choose Yes if your maps need optional cleaning or conditioning before figure export. "
            "You can mask background or pores using a greyscale image, protect very small values before ratio calculations, "
            "or apply a conditional rule such as setting Cl to zero where Fe is above a chosen threshold. "
            "Choose No if you only want direct maps, ratios, custom formulas, and normal visualization without these extra processing rules."
        )
        summary_label.setWordWrap(True)
        summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card_layout.addWidget(summary_label)

        choice_row = QHBoxLayout()
        choice_row.setSpacing(10)
        choice_row.addWidget(QLabel("Apply masking and noise removal:"))
        self.enable_combo = QComboBox()
        self.enable_combo.addItems(["No", "Yes"])
        self.enable_combo.setMinimumWidth(110)
        choice_row.addWidget(self.enable_combo)
        choice_row.addStretch()
        card_layout.addLayout(choice_row)

        layout.addWidget(card, 0, Qt.AlignTop)
        layout.addStretch()

        set_page_layout(self, layout, scrollable=False)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.enable_combo.setCurrentText(
            "Yes" if self.state.masking_and_noise_settings.enabled else "No"
        )

    def validatePage(self):
        self.state.masking_and_noise_settings.enabled = (self.enable_combo.currentText() == "Yes")
        return True

    def nextId(self):
        if self.enable_combo.currentText() == "Yes":
            return ElementtiWizard.PAGE_MASKING_SETTINGS
        return ElementtiWizard.PAGE_DISPLAY


class MaskingSettingsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 6: Masking and noise-removal settings")
        self.setSubTitle("Configure optional rules.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Set optional masking/noise rules. All thresholds are interpreted after scaling.",
            "Masking settings help",
            "Use this page to configure optional masking and noise-removal rules. You may use only one method or combine several methods in the same run. All threshold values, limits, and replacement values are interpreted after the scale factors from the preprocessing page have been applied. For example, if Fe values were scaled by 0.01, then a threshold of 70 means 70 after scaling, not 7000 in the original CSV.\n\n"
            "Method 1, greyscale masking, is useful when a greyscale or reference image identifies regions that should not be visually interpreted, such as empty background, pores, cracks, shadowed regions, or areas outside the sample. Greyscale masking hides pixels in figures and affects automatic visible min/max values, but it does not modify the processed numerical arrays saved as CSV.\n\n"
            "Method 2, low-value protection, replaces finite values that are less than or equal to a chosen minimum allowed value. This can reduce unstable ratios caused by very small denominator values. Because it changes the processed numerical array for the selected file, it can affect exported processed CSV files, ratio maps, and custom formula maps.\n\n"
            "Method 3, conditional replacement, modifies a selected target array before ratio calculation. For example, you can define rules such as: if Fe is above 70, set Cl = 0. The condition-source file and target file must have the same array shape. Use this only when the rule has a clear physical or analytical meaning for the dataset."
        )

        layout.addWidget(QLabel("Method 1: Grayscale masking"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Use this method:"))
        self.grayscale_enable_combo = QComboBox()
        self.grayscale_enable_combo.addItems(["No", "Yes"])
        self.grayscale_enable_combo.currentTextChanged.connect(self.update_enabled_state)
        row.addWidget(self.grayscale_enable_combo)
        row.addStretch()
        layout.addLayout(row)

        self.grayscale_table = QTableWidget(0, 7)
        self.grayscale_table.setHorizontalHeaderLabels(
            [
                "Use rule",
                "Source file",
                "Apply to output",
                "Condition",
                "Value A",
                "Value B",
                "Mask color",
            ]
        )
        self.grayscale_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.grayscale_table.verticalHeader().setVisible(False)
        layout.addWidget(self.grayscale_table)

        row = QHBoxLayout()
        self.add_grayscale_rule_button = QPushButton("Add grayscale rule")
        self.remove_grayscale_rule_button = QPushButton("Remove selected grayscale rule")
        self.add_grayscale_rule_button.clicked.connect(self.add_grayscale_rule)
        self.remove_grayscale_rule_button.clicked.connect(self.remove_selected_grayscale_rule)
        row.addWidget(self.add_grayscale_rule_button)
        row.addWidget(self.remove_grayscale_rule_button)
        row.addStretch()
        layout.addLayout(row)

        layout.addWidget(QLabel("Method 2: Low-value protection"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Use this method:"))
        self.low_value_enable_combo = QComboBox()
        self.low_value_enable_combo.addItems(["No", "Yes"])
        self.low_value_enable_combo.currentTextChanged.connect(self.update_enabled_state)
        row.addWidget(self.low_value_enable_combo)
        row.addStretch()
        layout.addLayout(row)

        self.low_value_table = QTableWidget(0, 3)
        self.low_value_table.setHorizontalHeaderLabels(
            ["File", "Use protection", "Minimum allowed value (after scaling)"]
        )
        self.low_value_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.low_value_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.low_value_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.low_value_table.verticalHeader().setVisible(False)
        layout.addWidget(self.low_value_table)

        layout.addWidget(QLabel("Method 3: Conditional replacement"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Use this method:"))
        self.conditional_enable_combo = QComboBox()
        self.conditional_enable_combo.addItems(["No", "Yes"])
        self.conditional_enable_combo.currentTextChanged.connect(self.update_enabled_state)
        row.addWidget(self.conditional_enable_combo)
        row.addStretch()
        layout.addLayout(row)

        self.conditional_table = QTableWidget(0, 7)
        self.conditional_table.setHorizontalHeaderLabels(
            [
                "Use rule",
                "If file",
                "Condition",
                "Value A",
                "Value B",
                "Then set file",
                "To value",
            ]
        )
        self.conditional_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.conditional_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.conditional_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.conditional_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.conditional_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.conditional_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.conditional_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.conditional_table.verticalHeader().setVisible(False)
        layout.addWidget(self.conditional_table)

        row = QHBoxLayout()
        self.add_rule_button = QPushButton("Add rule")
        self.remove_rule_button = QPushButton("Remove selected rule")
        self.add_rule_button.clicked.connect(self.add_conditional_rule)
        self.remove_rule_button.clicked.connect(self.remove_selected_conditional_rule)
        row.addWidget(self.add_rule_button)
        row.addWidget(self.remove_rule_button)
        row.addStretch()
        layout.addLayout(row)

        self.status_label = QLabel("Choose the masking and noise-removal settings you want, then click Next.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.populate_grayscale_table()
        self.populate_low_value_table()
        self.populate_conditional_table()

        settings = self.state.masking_and_noise_settings
        self.grayscale_enable_combo.setCurrentText("Yes" if settings.grayscale_enabled else "No")
        self.low_value_enable_combo.setCurrentText("Yes" if settings.low_value_enabled else "No")
        self.conditional_enable_combo.setCurrentText("Yes" if settings.conditional_enabled else "No")

        self.update_enabled_state()

    def ensure_default_low_value_settings_exist(self):
        settings = self.state.masking_and_noise_settings
        for name in self.state.get_ordered_names():
            settings.low_value_by_name.setdefault(name, LowValueRule())

    def populate_low_value_table(self):
        self.ensure_default_low_value_settings_exist()
        self.low_value_table.setRowCount(0)

        for name in self.state.get_ordered_names():
            row = self.low_value_table.rowCount()
            self.low_value_table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.low_value_table.setItem(row, 0, name_item)

            rule = self.state.masking_and_noise_settings.low_value_by_name[name]

            use_combo = QComboBox()
            use_combo.addItems(["No", "Yes"])
            use_combo.setCurrentText("Yes" if rule.enabled else "No")
            use_combo.currentTextChanged.connect(self.update_enabled_state)
            self.low_value_table.setCellWidget(row, 1, use_combo)

            floor_edit = QLineEdit(rule.floor)
            floor_edit.setPlaceholderText("Example: 0.05")
            self.low_value_table.setCellWidget(row, 2, floor_edit)

    def read_low_value_table_to_settings(self):
        result = {}
        for row in range(self.low_value_table.rowCount()):
            name = self.low_value_table.item(row, 0).text()
            use_combo = self.low_value_table.cellWidget(row, 1)
            floor_edit = self.low_value_table.cellWidget(row, 2)
            result[name] = LowValueRule(
                enabled=(use_combo.currentText() == "Yes"),
                floor=floor_edit.text().strip(),
            )
        self.state.masking_and_noise_settings.low_value_by_name = result

    def populate_grayscale_table(self):
        self.grayscale_table.setRowCount(0)
        names = self.state.get_ordered_names()
        output_names = self.state.get_output_names()
        rules = self.state.masking_and_noise_settings.grayscale_rules

        for rule in rules:
            self.add_grayscale_rule_row(rule, names, output_names)

    def add_grayscale_rule_row(
        self,
        rule: Optional[GrayscaleMaskRule] = None,
        names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        if names is None:
            names = self.state.get_ordered_names()
        if output_names is None:
            output_names = self.state.get_output_names()

        if rule is None:
            source_default = names[0] if names else ""
            rule = GrayscaleMaskRule(
                enabled=True,
                source_name=source_default,
                target_output_name="All outputs",
                condition="below",
                value_a="",
                value_b="",
                color="white",
            )

        row = self.grayscale_table.rowCount()
        self.grayscale_table.insertRow(row)

        use_combo = QComboBox()
        use_combo.addItems(["No", "Yes"])
        use_combo.setCurrentText("Yes" if rule.enabled else "No")
        use_combo.currentTextChanged.connect(lambda _text, r=row: self.update_grayscale_row_state(r))
        self.grayscale_table.setCellWidget(row, 0, use_combo)

        source_combo = QComboBox()
        source_combo.addItems(names)
        if rule.source_name:
            idx = source_combo.findText(rule.source_name)
            if idx >= 0:
                source_combo.setCurrentIndex(idx)
        self.grayscale_table.setCellWidget(row, 1, source_combo)

        target_combo = QComboBox()
        target_combo.addItems(["All outputs"] + output_names)
        target_name = rule.target_output_name.strip() or "All outputs"
        if target_combo.findText(target_name) < 0:
            target_combo.addItem(target_name)
        target_combo.setCurrentText(target_name)
        self.grayscale_table.setCellWidget(row, 2, target_combo)

        condition_combo = QComboBox()
        condition_combo.addItems(GRAYSCALE_CONDITIONS)
        condition_combo.setCurrentText(normalize_grayscale_condition(rule.condition) or "below")
        condition_combo.currentTextChanged.connect(lambda _text, r=row: self.update_grayscale_row_state(r))
        self.grayscale_table.setCellWidget(row, 3, condition_combo)

        value_a_edit = QLineEdit(rule.value_a)
        value_a_edit.setPlaceholderText("Example: 20")
        self.grayscale_table.setCellWidget(row, 4, value_a_edit)

        value_b_edit = QLineEdit(rule.value_b)
        value_b_edit.setPlaceholderText("Only for between/outside")
        self.grayscale_table.setCellWidget(row, 5, value_b_edit)

        color_edit = QLineEdit(rule.color)
        color_edit.setPlaceholderText("Example: black or #000000")
        self.grayscale_table.setCellWidget(row, 6, color_edit)

        self.update_grayscale_row_state(row)

    def read_grayscale_table_to_settings(self):
        rules = []
        for row in range(self.grayscale_table.rowCount()):
            use_combo = self.grayscale_table.cellWidget(row, 0)
            source_combo = self.grayscale_table.cellWidget(row, 1)
            target_combo = self.grayscale_table.cellWidget(row, 2)
            condition_combo = self.grayscale_table.cellWidget(row, 3)
            value_a_edit = self.grayscale_table.cellWidget(row, 4)
            value_b_edit = self.grayscale_table.cellWidget(row, 5)
            color_edit = self.grayscale_table.cellWidget(row, 6)

            rules.append(
                GrayscaleMaskRule(
                    enabled=(use_combo.currentText() == "Yes"),
                    source_name=source_combo.currentText().strip(),
                    target_output_name=target_combo.currentText().strip(),
                    condition=normalize_grayscale_condition(condition_combo.currentText().strip()),
                    value_a=value_a_edit.text().strip(),
                    value_b=value_b_edit.text().strip(),
                    color=color_edit.text().strip(),
                )
            )

        self.state.masking_and_noise_settings.grayscale_rules = rules

    def add_grayscale_rule(self):
        self.read_grayscale_table_to_settings()
        names = self.state.get_ordered_names()
        source_default = names[0] if names else ""

        self.state.masking_and_noise_settings.grayscale_rules.append(
            GrayscaleMaskRule(
                enabled=True,
                source_name=source_default,
                target_output_name="All outputs",
                condition="below",
                value_a="",
                value_b="",
                color="white",
            )
        )
        self.populate_grayscale_table()
        self.update_enabled_state()

    def remove_selected_grayscale_rule(self):
        row = self.grayscale_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No grayscale rule selected", "Please select a grayscale rule first.")
            return

        self.read_grayscale_table_to_settings()
        del self.state.masking_and_noise_settings.grayscale_rules[row]
        self.populate_grayscale_table()
        self.update_enabled_state()

    def update_grayscale_row_state(self, row: int):
        if row < 0 or row >= self.grayscale_table.rowCount():
            return

        method_enabled = self.grayscale_enable_combo.currentText() == "Yes"
        use_combo = self.grayscale_table.cellWidget(row, 0)
        source_combo = self.grayscale_table.cellWidget(row, 1)
        target_combo = self.grayscale_table.cellWidget(row, 2)
        condition_combo = self.grayscale_table.cellWidget(row, 3)
        value_a_edit = self.grayscale_table.cellWidget(row, 4)
        value_b_edit = self.grayscale_table.cellWidget(row, 5)
        color_edit = self.grayscale_table.cellWidget(row, 6)

        rule_enabled = method_enabled and use_combo.currentText() == "Yes"
        needs_two_values = condition_combo.currentText() in ("between", "outside")

        use_combo.setEnabled(method_enabled)
        source_combo.setEnabled(rule_enabled)
        target_combo.setEnabled(rule_enabled)
        condition_combo.setEnabled(rule_enabled)
        value_a_edit.setEnabled(rule_enabled)
        value_b_edit.setEnabled(rule_enabled and needs_two_values)
        color_edit.setEnabled(rule_enabled)

    def populate_conditional_table(self):
        self.conditional_table.setRowCount(0)
        names = self.state.get_ordered_names()
        rules = self.state.masking_and_noise_settings.conditional_rules

        for rule in rules:
            self.add_conditional_rule_row(rule, names)

    def add_conditional_rule_row(self, rule: Optional[ConditionalReplacementRule] = None, names: Optional[List[str]] = None):
        if names is None:
            names = self.state.get_ordered_names()

        if rule is None:
            source_default = names[0] if names else ""
            target_default = names[0] if names else ""
            rule = ConditionalReplacementRule(
                enabled=True,
                condition_source_name=source_default,
                condition_operator="above",
                value_a="",
                value_b="",
                target_name=target_default,
                replacement_value="",
            )

        row = self.conditional_table.rowCount()
        self.conditional_table.insertRow(row)

        use_combo = QComboBox()
        use_combo.addItems(["No", "Yes"])
        use_combo.setCurrentText("Yes" if rule.enabled else "No")
        use_combo.currentTextChanged.connect(lambda _text, r=row: self.update_conditional_row_state(r))
        self.conditional_table.setCellWidget(row, 0, use_combo)

        source_combo = QComboBox()
        source_combo.addItems(names)
        if rule.condition_source_name:
            idx = source_combo.findText(rule.condition_source_name)
            if idx >= 0:
                source_combo.setCurrentIndex(idx)
        self.conditional_table.setCellWidget(row, 1, source_combo)

        operator_combo = QComboBox()
        operator_combo.addItems(CONDITIONAL_OPERATORS)
        operator_combo.setCurrentText(normalize_conditional_operator(rule.condition_operator) or "above")
        operator_combo.currentTextChanged.connect(lambda _text, r=row: self.update_conditional_row_state(r))
        self.conditional_table.setCellWidget(row, 2, operator_combo)

        value_a_edit = QLineEdit(rule.value_a)
        value_a_edit.setPlaceholderText("Example: 90")
        self.conditional_table.setCellWidget(row, 3, value_a_edit)

        value_b_edit = QLineEdit(rule.value_b)
        value_b_edit.setPlaceholderText("Only for between")
        self.conditional_table.setCellWidget(row, 4, value_b_edit)

        target_combo = QComboBox()
        target_combo.addItems(names)
        if rule.target_name:
            idx = target_combo.findText(rule.target_name)
            if idx >= 0:
                target_combo.setCurrentIndex(idx)
        self.conditional_table.setCellWidget(row, 5, target_combo)

        replacement_edit = QLineEdit(rule.replacement_value)
        replacement_edit.setPlaceholderText("Example: 0")
        self.conditional_table.setCellWidget(row, 6, replacement_edit)

        self.update_conditional_row_state(row)

    def read_conditional_table_to_settings(self):
        rules = []
        for row in range(self.conditional_table.rowCount()):
            use_combo = self.conditional_table.cellWidget(row, 0)
            source_combo = self.conditional_table.cellWidget(row, 1)
            operator_combo = self.conditional_table.cellWidget(row, 2)
            value_a_edit = self.conditional_table.cellWidget(row, 3)
            value_b_edit = self.conditional_table.cellWidget(row, 4)
            target_combo = self.conditional_table.cellWidget(row, 5)
            replacement_edit = self.conditional_table.cellWidget(row, 6)

            rules.append(
                ConditionalReplacementRule(
                    enabled=(use_combo.currentText() == "Yes"),
                    condition_source_name=source_combo.currentText().strip(),
                    condition_operator=normalize_conditional_operator(operator_combo.currentText().strip()),
                    value_a=value_a_edit.text().strip(),
                    value_b=value_b_edit.text().strip(),
                    target_name=target_combo.currentText().strip(),
                    replacement_value=replacement_edit.text().strip(),
                )
            )

        self.state.masking_and_noise_settings.conditional_rules = rules

    def add_conditional_rule(self):
        self.read_conditional_table_to_settings()
        names = self.state.get_ordered_names()
        source_default = names[0] if names else ""
        target_default = names[0] if names else ""

        self.state.masking_and_noise_settings.conditional_rules.append(
            ConditionalReplacementRule(
                enabled=True,
                condition_source_name=source_default,
                condition_operator="above",
                value_a="",
                value_b="",
                target_name=target_default,
                replacement_value="",
            )
        )
        self.populate_conditional_table()
        self.update_enabled_state()

    def remove_selected_conditional_rule(self):
        row = self.conditional_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No rule selected", "Please select a rule first.")
            return

        self.read_conditional_table_to_settings()
        del self.state.masking_and_noise_settings.conditional_rules[row]
        self.populate_conditional_table()
        self.update_enabled_state()

    def update_conditional_row_state(self, row: int):
        if row < 0 or row >= self.conditional_table.rowCount():
            return

        method_enabled = self.conditional_enable_combo.currentText() == "Yes"
        use_combo = self.conditional_table.cellWidget(row, 0)
        source_combo = self.conditional_table.cellWidget(row, 1)
        operator_combo = self.conditional_table.cellWidget(row, 2)
        value_a_edit = self.conditional_table.cellWidget(row, 3)
        value_b_edit = self.conditional_table.cellWidget(row, 4)
        target_combo = self.conditional_table.cellWidget(row, 5)
        replacement_edit = self.conditional_table.cellWidget(row, 6)

        rule_enabled = method_enabled and use_combo.currentText() == "Yes"
        is_between = operator_combo.currentText() == "between"

        use_combo.setEnabled(method_enabled)
        source_combo.setEnabled(rule_enabled)
        operator_combo.setEnabled(rule_enabled)
        value_a_edit.setEnabled(rule_enabled)
        value_b_edit.setEnabled(rule_enabled and is_between)
        target_combo.setEnabled(rule_enabled)
        replacement_edit.setEnabled(rule_enabled)

    def update_enabled_state(self, *_args):
        grayscale_enabled = self.grayscale_enable_combo.currentText() == "Yes"
        low_value_enabled = self.low_value_enable_combo.currentText() == "Yes"
        conditional_enabled = self.conditional_enable_combo.currentText() == "Yes"

        self.grayscale_table.setEnabled(grayscale_enabled)
        self.add_grayscale_rule_button.setEnabled(grayscale_enabled)
        self.remove_grayscale_rule_button.setEnabled(grayscale_enabled)
        for row in range(self.grayscale_table.rowCount()):
            self.update_grayscale_row_state(row)

        self.low_value_table.setEnabled(low_value_enabled)
        for row in range(self.low_value_table.rowCount()):
            use_combo = self.low_value_table.cellWidget(row, 1)
            floor_edit = self.low_value_table.cellWidget(row, 2)
            use_combo.setEnabled(low_value_enabled)
            floor_edit.setEnabled(low_value_enabled and use_combo.currentText() == "Yes")

        self.conditional_table.setEnabled(conditional_enabled)
        self.add_rule_button.setEnabled(conditional_enabled)
        self.remove_rule_button.setEnabled(conditional_enabled)
        for row in range(self.conditional_table.rowCount()):
            self.update_conditional_row_state(row)

    def validatePage(self):
        self.read_grayscale_table_to_settings()
        self.read_low_value_table_to_settings()
        self.read_conditional_table_to_settings()

        settings = self.state.masking_and_noise_settings

        grayscale_enabled = self.grayscale_enable_combo.currentText() == "Yes"
        low_value_enabled = self.low_value_enable_combo.currentText() == "Yes"
        conditional_enabled = self.conditional_enable_combo.currentText() == "Yes"

        if not grayscale_enabled and not low_value_enabled and not conditional_enabled:
            QMessageBox.warning(
                self,
                "No method selected",
                "Please turn on at least one method, or go back and choose 'No' for masking and noise removal."
            )
            return False

        if grayscale_enabled:
            active_grayscale_rules = [rule for rule in settings.grayscale_rules if rule.enabled]
            if not active_grayscale_rules:
                QMessageBox.warning(
                    self,
                    "No grayscale rule",
                    "Grayscale masking is enabled, but there is no active grayscale rule."
                )
                return False

            available_targets = set(self.state.get_output_names()) | {"All outputs"}

            for idx, rule in enumerate(active_grayscale_rules, start=1):
                condition = normalize_grayscale_condition(rule.condition)

                if rule.source_name.strip() == "":
                    QMessageBox.warning(self, "Missing grayscale source", f"Grayscale rule {idx}: please choose the source file.")
                    return False

                if rule.target_output_name.strip() == "":
                    QMessageBox.warning(
                        self,
                        "Missing target output",
                        f"Grayscale rule {idx}: please choose which output this mask should apply to."
                    )
                    return False

                if rule.target_output_name.strip() not in available_targets:
                    QMessageBox.warning(
                        self,
                        "Invalid target output",
                        f"Grayscale rule {idx}: target output '{rule.target_output_name}' is not available."
                    )
                    return False

                if rule.color.strip() == "":
                    QMessageBox.warning(self, "Missing mask color", f"Grayscale rule {idx}: please enter the mask color.")
                    return False

                if not is_color_like(rule.color.strip()):
                    QMessageBox.warning(
                        self,
                        "Invalid mask color",
                        f"Grayscale rule {idx}: '{rule.color}' is not a valid color."
                    )
                    return False

                if condition in ("below", "above"):
                    if rule.value_a.strip() == "":
                        QMessageBox.warning(self, "Missing threshold", f"Grayscale rule {idx}: please enter Value A.")
                        return False
                    try:
                        parse_float_user_text(rule.value_a, "Value A")
                    except ValueError:
                        QMessageBox.warning(self, "Invalid threshold", f"Grayscale rule {idx}: Value A must be a number.")
                        return False

                elif condition in ("between", "outside"):
                    if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                        QMessageBox.warning(
                            self,
                            "Missing range values",
                            f"Grayscale rule {idx}: '{condition}' requires both Value A and Value B."
                        )
                        return False
                    try:
                        parse_float_user_text(rule.value_a, "Value A")
                        parse_float_user_text(rule.value_b, "Value B")
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            "Invalid range values",
                            f"Grayscale rule {idx}: Value A and Value B must be numbers."
                        )
                        return False
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid grayscale condition",
                        f"Grayscale rule {idx}: '{rule.condition}' is not a valid condition."
                    )
                    return False

        if low_value_enabled:
            any_row_enabled = False
            for name, rule in settings.low_value_by_name.items():
                if not rule.enabled:
                    continue

                any_row_enabled = True
                floor_text = rule.floor.strip()

                if floor_text == "":
                    QMessageBox.warning(self, "Missing minimum value", f"Please enter the minimum allowed value for '{name}'.")
                    return False
                try:
                    floor_value = parse_float_user_text(floor_text, "Low-value floor")
                    if floor_value < 0:
                        raise ValueError
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid minimum value",
                        f"The minimum allowed value for '{name}' must be a number greater than or equal to 0."
                    )
                    return False

            if not any_row_enabled:
                QMessageBox.warning(
                    self,
                    "No file selected",
                    "Low-value protection is enabled, but no file is selected for protection."
                )
                return False

        if conditional_enabled:
            enabled_rules = [rule for rule in settings.conditional_rules if rule.enabled]
            if not enabled_rules:
                QMessageBox.warning(
                    self,
                    "No conditional rule",
                    "Conditional replacement is enabled, but there is no active rule."
                )
                return False

            for idx, rule in enumerate(enabled_rules, start=1):
                operator = normalize_conditional_operator(rule.condition_operator)

                if rule.condition_source_name.strip() == "":
                    QMessageBox.warning(self, "Missing source file", f"Rule {idx}: please choose the source file.")
                    return False
                if rule.target_name.strip() == "":
                    QMessageBox.warning(self, "Missing target file", f"Rule {idx}: please choose the target file.")
                    return False
                if rule.replacement_value.strip() == "":
                    QMessageBox.warning(self, "Missing replacement value", f"Rule {idx}: please enter the replacement value.")
                    return False
                try:
                    parse_float_user_text(rule.replacement_value, "Replacement value")
                except ValueError:
                    QMessageBox.warning(self, "Invalid replacement value", f"Rule {idx}: replacement value must be a number.")
                    return False

                if operator == "between":
                    if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                        QMessageBox.warning(self, "Missing condition values", f"Rule {idx}: 'between' requires both Value A and Value B.")
                        return False
                    try:
                        parse_float_user_text(rule.value_a, "Value A")
                        parse_float_user_text(rule.value_b, "Value B")
                    except ValueError:
                        QMessageBox.warning(self, "Invalid condition values", f"Rule {idx}: Value A and Value B must be numbers.")
                        return False
                elif operator in ("above", "below"):
                    if rule.value_a.strip() == "":
                        QMessageBox.warning(self, "Missing condition value", f"Rule {idx}: please enter Value A.")
                        return False
                    try:
                        parse_float_user_text(rule.value_a, "Value A")
                    except ValueError:
                        QMessageBox.warning(self, "Invalid condition value", f"Rule {idx}: Value A must be a number.")
                        return False
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid condition",
                        f"Rule {idx}: '{rule.condition_operator}' is not a valid condition."
                    )
                    return False

        settings.enabled = True
        settings.grayscale_enabled = grayscale_enabled
        settings.low_value_enabled = low_value_enabled
        settings.conditional_enabled = conditional_enabled

        return True


class DisplaySettingsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 7: Choose display settings")
        self.setSubTitle("Set visualization parameters.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Choose colormaps, limits, colorbar labels, and optional manual bins.",
            "Display settings help",
            "Use this page to control how each selected output will appear in the exported figures. Each row corresponds to one direct map, ratio map, or custom formula map selected earlier.\n\n"
            "Continuous mode displays the map using a Matplotlib colormap and a minimum/maximum range. If display min or max is left blank, Elementti uses the processed visible minimum or maximum calculated for that output. If values are clipped by a manual display min or max, the colorbar endpoint is labeled with ≤ or ≥, for example ≥21, so the figure makes clear that values at and above the chosen maximum share the top color.\n\n"
            "Manual-bin mode groups values into user-defined intervals. Enter boundary values separated by commas, for example 0, 0.5, 1, 1.5, 2. If you use decimal commas, separate bin boundaries with semicolons, for example 0; 0,5; 1,0. You must provide one fewer color than the number of boundaries, because each interval needs one color.\n\n"
            "Colors can be common color names such as white, black, gray, red, orange, yellow, green, cyan, blue, purple, magenta, or brown. Hex colors are also accepted, for example #FFFFFF, #000000, #FF0000, #00FF00, #0000FF, and #FFA500.\n\n"
            "Colorbar labels and colorbar side can be adjusted for each output. These settings are saved in the JSON summary so the figure preparation is traceable.\n\n"
            "This page also lets you choose whether Elementti should export additional figures with physical scale bars. Pixel sizes are entered per input file in Step 4 and stored internally as µm/pixel. Step 4 can auto-convert clear metadata or typed entries using nm, µm/um, mm, cm, or m per pixel. If a scale-bar length is left blank, Elementti automatically chooses a clean rounded length close to one fifth to one fourth of each map width, using readable labels such as 500 nm, 20 µm, or 1 mm. If you enter a manual length, it must be large enough to be visible and not too large for the image."
        )

        scale_row_1 = QHBoxLayout()
        scale_row_1.addWidget(QLabel("Export additional figures with scale bar:"))
        self.scale_bar_enable_combo = QComboBox()
        self.scale_bar_enable_combo.addItems(["No", "Yes"])
        self.scale_bar_enable_combo.currentTextChanged.connect(self.update_scale_bar_controls)
        self.scale_bar_enable_combo.currentTextChanged.connect(self.schedule_preview_update)
        scale_row_1.addWidget(self.scale_bar_enable_combo)
        scale_row_1.addSpacing(10)
        scale_row_1.addWidget(QLabel("Scale-bar length (µm, optional):"))
        self.scale_bar_length_edit = QLineEdit()
        self.scale_bar_length_edit.setPlaceholderText("blank = automatic, e.g. 50")
        self.scale_bar_length_edit.setMaximumWidth(170)
        self.scale_bar_length_edit.textChanged.connect(self.schedule_preview_update)
        scale_row_1.addWidget(self.scale_bar_length_edit)
        scale_row_1.addSpacing(10)
        scale_row_1.addWidget(QLabel("Color:"))
        self.scale_bar_color_combo = QComboBox()
        self.scale_bar_color_combo.addItems(["white", "black", "yellow", "red", "blue"])
        self.scale_bar_color_combo.currentTextChanged.connect(self.schedule_preview_update)
        scale_row_1.addWidget(self.scale_bar_color_combo)
        scale_row_1.addSpacing(10)
        scale_row_1.addWidget(QLabel("Position:"))
        self.scale_bar_position_combo = QComboBox()
        self.scale_bar_position_combo.addItems(["lower right", "lower left", "upper right", "upper left"])
        self.scale_bar_position_combo.currentTextChanged.connect(self.schedule_preview_update)
        scale_row_1.addWidget(self.scale_bar_position_combo)
        scale_row_1.addStretch()
        layout.addLayout(scale_row_1)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Output",
                "Mode",
                "Colormap",
                "Guide bar side",
                "Guide bar label",
                "Processed min",
                "Processed max",
                "Display min",
                "Display max",
                "Bins",
                "Bin colors",
            ]
        )
        for col in range(11):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(10, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.schedule_preview_update)
        layout.addWidget(self.table)

        self.preview_title_label = QLabel("Live preview")
        self.preview_title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.preview_title_label)

        self.preview_figure = Figure(figsize=(8.0, 3.8), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.preview_canvas.setMinimumHeight(340)
        self.preview_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_canvas, stretch=1)

        self.status_label = QLabel("Choose the display settings you want, then click Next.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.raw_arrays_cache = {}
        self.working_arrays_cache = {}
        self.output_arrays_cache = {}
        self._preview_suppress = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(180)
        self.preview_timer.timeout.connect(self.draw_selected_output_preview)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 7: Choose display settings")
        else:
            self.setTitle("Step 6: Choose display settings")

        sb = self.state.scale_bar_settings
        self.scale_bar_enable_combo.setCurrentText("Yes" if sb.enabled else "No")
        self.scale_bar_length_edit.setText(sb.length_um)
        self.scale_bar_color_combo.setCurrentText(sb.color if sb.color else "white")
        self.scale_bar_position_combo.setCurrentText(sb.position if sb.position else "lower right")
        self.update_scale_bar_controls()

        try:
            self.refresh_preview_caches()
            self.populate_table()
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
            self.draw_selected_output_preview()
            self.status_label.setText("Processed min/max were calculated successfully.")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Could not calculate processed min/max",
                f"The app could not read one of the files with the current settings.\n\n{str(e)}"
            )
            self.preview_figure.clear()
            self.preview_canvas.draw_idle()
            self.status_label.setText("There was a problem reading one of the files. Please go back and check earlier steps.")

    def get_output_names(self) -> List[str]:
        return self.state.get_output_names()

    def default_colormap_for_output(self, output_name: str) -> str:
        if "/" in output_name:
            return "plasma"
        lower = output_name.lower()
        if lower == "fe":
            return "inferno"
        if lower == "o":
            return "viridis"
        if is_grayscale_name(output_name):
            return "gray"
        return "viridis"

    def ensure_default_display_settings_exist(self):
        self.state.prune_display_settings_to_current_outputs()

        for output_name in self.get_output_names():
            if output_name in self.state.display_settings:
                continue

            self.state.display_settings[output_name] = OutputDisplaySettings(
                mode="Continuous",
                colormap=self.default_colormap_for_output(output_name),
                colorbar_side="right",
                colorbar_label=default_colorbar_label(output_name),
                display_min="",
                display_max="",
            )

    def populate_table(self):
        self.wizard().engine.compute_output_stats()
        self.ensure_default_display_settings_exist()
        self.table.setRowCount(0)

        for output_name in self.get_output_names():
            row = self.table.rowCount()
            self.table.insertRow(row)

            settings = self.state.display_settings[output_name]
            min_val, max_val = self.state.processed_min_max_by_name.get(output_name, (None, None))

            output_item = QTableWidgetItem(output_name)
            output_item.setFlags(output_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, output_item)

            mode_combo = QComboBox()
            mode_combo.addItems(["Continuous", "Manual bins"])
            mode_combo.setCurrentText(settings.mode)
            mode_combo.currentTextChanged.connect(lambda _text, r=row: self.on_mode_changed(r))
            mode_combo.currentTextChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 1, mode_combo)

            cmap_combo = QComboBox()
            cmap_combo.addItems([
                "viridis",
                "inferno",
                "plasma",
                "magma",
                "cividis",
                "gray",
                "hot",
                "jet",
                "green",
                "red",
                "blue",
                "yellow",
                "orange",
                "purple",
                "cyan",
                "brown",
            ])
            cmap_combo.setCurrentText(settings.colormap)
            cmap_combo.currentTextChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 2, cmap_combo)

            side_combo = QComboBox()
            side_combo.addItems(["right", "left"])
            side_combo.setCurrentText(settings.colorbar_side)
            side_combo.currentTextChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 3, side_combo)

            colorbar_label_edit = QLineEdit(settings.colorbar_label)
            colorbar_label_edit.textChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 4, colorbar_label_edit)

            min_item = QTableWidgetItem("" if min_val is None else f"{min_val:g}")
            min_item.setFlags(min_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, min_item)

            max_item = QTableWidgetItem("" if max_val is None else f"{max_val:g}")
            max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, max_item)

            display_min_edit = QLineEdit(settings.display_min)
            display_min_edit.setPlaceholderText("use processed min")
            display_min_edit.textChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 7, display_min_edit)

            display_max_edit = QLineEdit(settings.display_max)
            display_max_edit.setPlaceholderText("use processed max")
            display_max_edit.textChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 8, display_max_edit)

            bins_edit = QLineEdit(settings.bins)
            bins_edit.setPlaceholderText("Example: 0, 0.5, 1, 1.5, 2")
            bins_edit.textChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 9, bins_edit)

            bin_colors_edit = QLineEdit(settings.bin_colors)
            bin_colors_edit.setPlaceholderText("Example: blue, green, yellow, red or #FF0000")
            bin_colors_edit.textChanged.connect(self.schedule_preview_update)
            self.table.setCellWidget(row, 10, bin_colors_edit)

            self.on_mode_changed(row)

    def on_mode_changed(self, row):
        mode_combo = self.table.cellWidget(row, 1)
        cmap_combo = self.table.cellWidget(row, 2)
        bins_edit = self.table.cellWidget(row, 9)
        bin_colors_edit = self.table.cellWidget(row, 10)

        if mode_combo.currentText() == "Manual bins":
            bins_edit.setEnabled(True)
            bin_colors_edit.setEnabled(True)
            cmap_combo.setEnabled(False)
        else:
            bins_edit.setEnabled(False)
            bin_colors_edit.setEnabled(False)
            cmap_combo.setEnabled(True)

    def refresh_preview_caches(self):
        self.raw_arrays_cache = self.wizard().engine.load_raw_arrays()
        self.working_arrays_cache = self.wizard().engine.apply_processing_rules(self.raw_arrays_cache)
        self.output_arrays_cache = self.wizard().engine.build_output_arrays(
            self.raw_arrays_cache,
            self.working_arrays_cache,
        )

    def schedule_preview_update(self, *_args):
        if self._preview_suppress:
            return
        self.preview_timer.start()

    def current_preview_row(self) -> int:
        row = self.table.currentRow()
        if row < 0 and self.table.rowCount() > 0:
            row = 0
        return row

    def build_display_settings_from_row(self, row: int) -> Optional[OutputDisplaySettings]:
        if row < 0 or row >= self.table.rowCount():
            return None
        output_name = self.table.item(row, 0).text().strip()
        return OutputDisplaySettings(
            mode=self.table.cellWidget(row, 1).currentText().strip(),
            colormap=self.table.cellWidget(row, 2).currentText().strip(),
            colorbar_side=self.table.cellWidget(row, 3).currentText().strip(),
            colorbar_label=self.table.cellWidget(row, 4).text().strip() or default_colorbar_label(output_name),
            display_min=self.table.cellWidget(row, 7).text().strip(),
            display_max=self.table.cellWidget(row, 8).text().strip(),
            bins=self.table.cellWidget(row, 9).text().strip(),
            bin_colors=self.table.cellWidget(row, 10).text().strip(),
        )

    def current_scale_bar_preview_settings(self) -> ScaleBarSettings:
        return ScaleBarSettings(
            enabled=(self.scale_bar_enable_combo.currentText() == "Yes"),
            length_um=self.scale_bar_length_edit.text().strip(),
            color=self.scale_bar_color_combo.currentText().strip(),
            position=self.scale_bar_position_combo.currentText().strip(),
        )

    def draw_selected_output_preview(self):
        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111)
        row = self.current_preview_row()
        if row < 0:
            ax.text(0.5, 0.5, "No output selected", ha="center", va="center")
            ax.set_axis_off()
            self.preview_title_label.setText("Live preview")
            self.preview_canvas.draw_idle()
            return

        output_name = self.table.item(row, 0).text().strip()
        self.preview_title_label.setText(f"Live preview: {output_name}")

        if output_name not in self.output_arrays_cache:
            try:
                self.refresh_preview_caches()
            except Exception as e:
                ax.text(0.5, 0.5, f"Preview unavailable:\n{str(e)}", ha="center", va="center", wrap=True)
                ax.set_axis_off()
                self.preview_canvas.draw_idle()
                return

        if output_name not in self.output_arrays_cache:
            available = ", ".join(self.output_arrays_cache.keys()) if self.output_arrays_cache else "none"
            ax.text(
                0.5,
                0.5,
                f"Preview unavailable for '{output_name}'.\nAvailable outputs: {available}",
                ha="center",
                va="center",
                wrap=True,
            )
            ax.set_axis_off()
            self.preview_canvas.draw_idle()
            return

        settings = self.build_display_settings_from_row(row)
        array = self.output_arrays_cache[output_name]
        mask_layers = self.wizard().engine.build_grayscale_mask_layers(self.raw_arrays_cache, output_name=output_name)

        pixel_size_um = None
        try:
            pixel_size_um = self.wizard().engine.get_pixel_size_um_for_output(output_name)
        except Exception:
            pixel_size_um = None

        old_scale_settings = self.state.scale_bar_settings
        self.state.scale_bar_settings = self.current_scale_bar_preview_settings()
        try:
            self.wizard().engine.plot_output_map_axis(
                ax,
                array,
                output_name,
                settings,
                mask_layers=mask_layers,
                line_coords=None,
                show_colorbar=True,
                add_scale_bar=self.state.scale_bar_settings.enabled and (pixel_size_um is not None),
                pixel_size_um=pixel_size_um,
            )
        except Exception as e:
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Preview unavailable:\n{str(e)}", ha="center", va="center", wrap=True)
            ax.set_axis_off()
        finally:
            self.state.scale_bar_settings = old_scale_settings

        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()

    def update_scale_bar_controls(self):
        enabled = (self.scale_bar_enable_combo.currentText() == "Yes")
        for widget in [
            self.scale_bar_length_edit,
            self.scale_bar_color_combo,
            self.scale_bar_position_combo,
        ]:
            widget.setEnabled(enabled)
        self.schedule_preview_update()

    def validate_number_or_blank(self, value_text: str, field_name: str, output_name: str) -> bool:
        if value_text.strip() == "":
            return True
        try:
            parse_float_user_text(value_text, field_name)
        except ValueError:
            QMessageBox.warning(self, "Invalid number", f"{field_name} for '{output_name}' must be a number or left blank.")
            return False
        return True

    def validate_bins(self, bins_text: str, output_name: str) -> bool:
        if bins_text.strip() == "":
            QMessageBox.warning(self, "Missing bins", f"Please enter manual bins for '{output_name}', or change its mode.")
            return False

        parts = split_numeric_list_text(bins_text)
        if len(parts) < 2:
            QMessageBox.warning(self, "Not enough bins", f"Manual bins for '{output_name}' must contain at least two numbers.")
            return False

        try:
            values = parse_bins_text(bins_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid bins", f"All manual bin boundaries for '{output_name}' must be numbers separated by commas.")
            return False

        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                QMessageBox.warning(self, "Invalid bins order", f"Manual bin boundaries for '{output_name}' must increase from left to right.")
                return False

        return True

    def validate_bin_colors(self, bin_colors_text: str, bins_text: str, output_name: str) -> bool:
        if bin_colors_text.strip() == "":
            QMessageBox.warning(self, "Missing bin colors", f"Please enter manual bin colors for '{output_name}', or change its mode.")
            return False

        colors = [c.strip() for c in bin_colors_text.split(",") if c.strip()]
        bins = parse_bins_text(bins_text)
        expected_color_count = len(bins) - 1

        if len(colors) != expected_color_count:
            QMessageBox.warning(
                self,
                "Wrong number of bin colors",
                f"For '{output_name}', you entered {len(colors)} colors but need {expected_color_count} colors.\n\n"
                f"If your boundaries are {bins_text}, then the bins are made between each neighboring pair.\n"
                f"So {len(bins)} boundary numbers need exactly {expected_color_count} colors."
            )
            return False

        for color in colors:
            if not is_color_like(color):
                QMessageBox.warning(
                    self,
                    "Invalid color",
                    f"'{color}' is not a valid color for '{output_name}'.\n\n"
                    f"Use names like blue, green, yellow, red, orange, purple, black, white, gray,\n"
                    f"or use hex codes like #FF0000."
                )
                return False

        return True

    def validatePage(self):
        output_names = self.get_output_names()
        if output_names and self.table.rowCount() == 0:
            QMessageBox.warning(
                self,
                "Display settings unavailable",
                "The display table is empty because the current files or processing settings could not be read."
            )
            return False

        result: Dict[str, OutputDisplaySettings] = {}

        for row in range(self.table.rowCount()):
            output_name = self.table.item(row, 0).text()

            mode = self.table.cellWidget(row, 1).currentText().strip()
            cmap = self.table.cellWidget(row, 2).currentText().strip()
            colorbar_side = self.table.cellWidget(row, 3).currentText().strip()
            colorbar_label = self.table.cellWidget(row, 4).text().strip()
            display_min_text = self.table.cellWidget(row, 7).text().strip()
            display_max_text = self.table.cellWidget(row, 8).text().strip()
            bins_text = self.table.cellWidget(row, 9).text().strip()
            bin_colors_text = self.table.cellWidget(row, 10).text().strip()

            if not self.validate_number_or_blank(display_min_text, "Display minimum", output_name):
                return False
            if not self.validate_number_or_blank(display_max_text, "Display maximum", output_name):
                return False

            if display_min_text and display_max_text and parse_float_user_text(display_min_text, "Display minimum") >= parse_float_user_text(display_max_text, "Display maximum"):
                QMessageBox.warning(self, "Invalid range", f"For '{output_name}', display minimum must be smaller than display maximum.")
                return False

            if mode == "Manual bins":
                if not self.validate_bins(bins_text, output_name):
                    return False
                if not self.validate_bin_colors(bin_colors_text, bins_text, output_name):
                    return False

            result[output_name] = OutputDisplaySettings(
                mode=mode,
                colormap=cmap,
                colorbar_side=colorbar_side,
                colorbar_label=colorbar_label or default_colorbar_label(output_name),
                display_min=display_min_text,
                display_max=display_max_text,
                bins=bins_text,
                bin_colors=bin_colors_text,
            )

        scale_enabled = (self.scale_bar_enable_combo.currentText() == "Yes")
        scale_bar_length_text = self.scale_bar_length_edit.text().strip()
        if scale_enabled:
            manual_scale_length_value = None
            if scale_bar_length_text != "":
                try:
                    manual_scale_length_value = parse_float_user_text(scale_bar_length_text, "Scale-bar length (µm)")
                    if manual_scale_length_value <= 0:
                        raise ValueError
                    scale_bar_length_text = format_detected_float(manual_scale_length_value)
                except ValueError:
                    QMessageBox.warning(self, "Invalid scale-bar length", "Scale-bar length (µm) must be a number greater than zero, or left blank for automatic length.")
                    return False

            missing_or_invalid = []
            for output_name in output_names:
                try:
                    pixel_size_value = self.wizard().engine.get_pixel_size_um_for_output(output_name)
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid scale-bar pixel size", str(e))
                    return False
                if pixel_size_value is None:
                    missing_or_invalid.append(output_name)
                    continue

                if manual_scale_length_value is not None:
                    shape = None
                    if output_name in self.state.processed_min_max_by_name:
                        try:
                            # Use loaded arrays only when needed for manual scale-bar validation.
                            raw_arrays = self.wizard().engine.load_raw_arrays()
                            working_arrays = self.wizard().engine.apply_processing_rules(raw_arrays)
                            output_arrays = self.wizard().engine.build_output_arrays(raw_arrays, working_arrays)
                            if output_name in output_arrays:
                                shape = output_arrays[output_name].shape
                        except Exception:
                            shape = None
                    if shape is not None:
                        rows, cols = shape
                        length_px = manual_scale_length_value / pixel_size_value
                        if length_px < 10:
                            QMessageBox.warning(
                                self,
                                "Scale bar too small",
                                f"The manual scale bar for '{output_name}' would be only {length_px:.1f} pixels long. "
                                "Choose a longer value, or leave the length blank so Elementti chooses an automatic clean value."
                            )
                            return False
                        if length_px > cols * 0.50:
                            QMessageBox.warning(
                                self,
                                "Scale bar too large",
                                f"The manual scale bar for '{output_name}' would be {length_px:.1f} pixels long, "
                                f"which is more than half of the image width ({cols} pixels). Choose a shorter value, "
                                "or leave the length blank so Elementti chooses an automatic clean value."
                            )
                            return False

            if missing_or_invalid:
                QMessageBox.warning(
                    self,
                    "Missing pixel size",
                    "Scale-bar export is enabled, but these selected outputs do not have usable pixel-size information:\n\n"
                    + "\n".join(f"- {name}" for name in missing_or_invalid)
                    + "\n\nPlease go back to Step 4 and enter Pixel size (µm/pixel), or choose No for scale-bar export."
                )
                return False

        self.state.display_settings = result
        self.state.scale_bar_settings = ScaleBarSettings(
            enabled=scale_enabled,
            length_um=scale_bar_length_text,
            color=self.scale_bar_color_combo.currentText().strip(),
            position=self.scale_bar_position_combo.currentText().strip(),
        )
        return True



class LineProfileIntroPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 8: Line profile / transect analysis")
        self.setSubTitle("Choose whether to create an optional line-profile figure.")

        layout = QVBoxLayout()
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("optionalDecisionCard")
        card.setMinimumWidth(760)
        card.setMaximumWidth(1100)
        card.setMinimumHeight(220)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.setStyleSheet(
            "QFrame#optionalDecisionCard { background: white; border: 1px solid #d6d6d6; border-radius: 8px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.setSpacing(14)

        title_label = QLabel("Optional line profile / transect")
        title_label.setStyleSheet("font-weight: 700; font-size: 15px;")
        card_layout.addWidget(title_label)

        summary_label = QLabel(
            "Choose Yes if you want to draw a line across one map or reference image and generate a two-panel figure: "
            "the left panel shows the selected map with the transect line, and the right panel shows the selected elemental or ratio values along that line. "
            "Choose No if you only want the normal exported maps, ratios, custom formulas, CSV files, and methods summary."
        )
        summary_label.setWordWrap(True)
        summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card_layout.addWidget(summary_label)

        choice_row = QHBoxLayout()
        choice_row.setSpacing(10)
        choice_row.addWidget(QLabel("Create line profile / transect:"))
        self.enable_combo = QComboBox()
        self.enable_combo.addItems(["No", "Yes"])
        self.enable_combo.setMinimumWidth(110)
        choice_row.addWidget(self.enable_combo)
        choice_row.addStretch()
        card_layout.addLayout(choice_row)

        layout.addWidget(card, 0, Qt.AlignTop)
        layout.addStretch()

        set_page_layout(self, layout, scrollable=False)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 8: Line profile / transect analysis")
        else:
            self.setTitle("Step 7: Line profile / transect analysis")
        self.enable_combo.setCurrentText("Yes" if self.state.line_profile_settings.enabled else "No")

    def validatePage(self):
        self.state.line_profile_settings.enabled = (self.enable_combo.currentText() == "Yes")
        return True

    def nextId(self):
        if self.enable_combo.currentText() == "Yes":
            return ElementtiWizard.PAGE_LINE_PROFILE
        return ElementtiWizard.PAGE_GENERATE


class LineProfilePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 9: Line-profile settings")
        self.setSubTitle("Draw a line and preview the combined two-panel figure.")

        self.raw_arrays_cache: Dict[str, np.ndarray] = {}
        self.output_arrays_cache: Dict[str, np.ndarray] = {}
        self.preview_point: Optional[Tuple[float, float]] = None
        self._updating_controls = False

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Compact top controls. Detailed instructions are moved into help popups so the page
        # remains usable on laptop screens.
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self.enable_combo = QComboBox()
        self.enable_combo.addItems(["No", "Yes"])
        self.enable_combo.currentTextChanged.connect(self.on_controls_changed)
        self.enable_combo.setVisible(False)

        top_row.addWidget(QLabel("Background:"))
        self.base_output_combo = QComboBox()
        self.base_output_combo.currentTextChanged.connect(self.on_base_output_changed)
        top_row.addWidget(self.base_output_combo, stretch=1)

        self.clear_line_button = QPushButton("Clear / redraw line")
        self.clear_line_button.clicked.connect(self.clear_line)
        top_row.addWidget(self.clear_line_button)

        top_row.addWidget(self.make_help_button(
            "Line profile help",
            "Use this optional page to extract a line profile, also called a transect, from one or more selected maps, ratio maps, or custom formula maps. A line profile shows how selected values change along a line drawn across the map.\n\n"
            "Choose the background output shown on the left. The background is used to help you position the line; the curves plotted on the right are selected separately in the Profiles tab.\n\n"
            "To draw the line, click once on the map to set the start point. Move the mouse to preview the line position, then click a second time to set the end point. The profile plot updates immediately after the second click. Click 'Clear / redraw line' if you want to choose a different transect.\n\n"
            "In the Profiles tab, select which outputs should be plotted, choose the curve colors, set line widths, and choose whether the plot should show raw values, normalized 0-1 values, or values normalized by maximum. Colors can be normal color names such as blue, orange, green, red, black, or hex codes such as #FF0000. Normalization affects only the plotted line-profile display; the exported CSV still stores the extracted numerical values.\n\n"
            "The line coordinates are shown in editable boxes under the preview. You can draw the line with the mouse, or type/change the start and end x-y coordinates manually. Coordinates are entered in pixels, but the profile x-axis automatically uses nm, µm, or mm when a pixel size is available. The exported CSV keeps both pixel distance and the plotted physical distance.\n\n"
            "In the Style tab, edit the optional left image title/x-label/guide-bar label and the line-profile title/labels. Leave the x-label blank for automatic physical-distance labeling. Optional left and right y-axis ranges let you zoom into a chosen value interval. When single-element curves and ratio-like curves are selected together, Elementti plots single-element values on the left y-axis and ratios on the right y-axis. If only ratios are selected, they are plotted on the left y-axis.\n\n"
            "By default, the legend is placed inside the plot with a small amount of reserved space so that it is less likely to cover the curves. In the Style tab, you can move the legend outside to the right as a vertical legend when many curves make the plot crowded."
        ))
        layout.addLayout(top_row)

        # Combined preview area. The live view matches the exported two-panel figure:
        # left = map with the selected transect, right = line-profile plot.
        self.combined_figure = Figure(figsize=(8.4, 3.4), dpi=100)
        self.combined_canvas = FigureCanvas(self.combined_figure)
        self.combined_canvas.setFixedSize(840, 340)
        self.combined_canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.map_ax = self.combined_figure.add_subplot(121)
        self.profile_ax = self.combined_figure.add_subplot(122)

        self.combined_canvas.mpl_connect("button_press_event", self.on_map_click)
        self.combined_canvas.mpl_connect("motion_notify_event", self.on_map_motion)
        layout.addWidget(self.combined_canvas, alignment=Qt.AlignCenter)

        coord_row = QHBoxLayout()
        coord_row.setSpacing(6)
        coord_row.addWidget(QLabel("Line coordinates (pixels):"))

        self.start_x_edit = QLineEdit()
        self.start_x_edit.setPlaceholderText("start x")
        self.start_x_edit.setMaximumWidth(80)
        self.start_x_edit.textChanged.connect(self.on_coordinate_text_changed)
        coord_row.addWidget(QLabel("Start x"))
        coord_row.addWidget(self.start_x_edit)

        self.start_y_edit = QLineEdit()
        self.start_y_edit.setPlaceholderText("start y")
        self.start_y_edit.setMaximumWidth(80)
        self.start_y_edit.textChanged.connect(self.on_coordinate_text_changed)
        coord_row.addWidget(QLabel("Start y"))
        coord_row.addWidget(self.start_y_edit)

        self.end_x_edit = QLineEdit()
        self.end_x_edit.setPlaceholderText("end x")
        self.end_x_edit.setMaximumWidth(80)
        self.end_x_edit.textChanged.connect(self.on_coordinate_text_changed)
        coord_row.addWidget(QLabel("End x"))
        coord_row.addWidget(self.end_x_edit)

        self.end_y_edit = QLineEdit()
        self.end_y_edit.setPlaceholderText("end y")
        self.end_y_edit.setMaximumWidth(80)
        self.end_y_edit.textChanged.connect(self.on_coordinate_text_changed)
        coord_row.addWidget(QLabel("End y"))
        coord_row.addWidget(self.end_y_edit)
        coord_row.addStretch()
        layout.addLayout(coord_row)

        # Compact tabbed controls. This prevents the page from becoming too tall on laptops.
        self.tabs = QTabWidget()

        profiles_tab = QWidget()
        profiles_layout = QVBoxLayout(profiles_tab)
        profiles_layout.setContentsMargins(6, 6, 6, 6)
        profiles_layout.setSpacing(5)

        profile_controls_row = QHBoxLayout()
        profile_controls_row.addWidget(QLabel("Normalization:"))
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["Raw values", "Normalize 0-1", "Normalize by maximum"])
        self.normalize_combo.currentTextChanged.connect(self.on_controls_changed)
        profile_controls_row.addWidget(self.normalize_combo)
        profile_controls_row.addStretch()
        profiles_layout.addLayout(profile_controls_row)

        self.curve_table = QTableWidget(0, 4)
        self.curve_table.setHorizontalHeaderLabels(["Output", "Plot", "Line color", "Line width"])
        self.curve_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.curve_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.setAlternatingRowColors(True)
        self.curve_table.setMinimumHeight(130)
        self.curve_table.itemChanged.connect(self.on_curve_table_changed)
        profiles_layout.addWidget(self.curve_table)
        self.tabs.addTab(profiles_tab, "Profiles")

        style_tab = QWidget()
        style_layout = QVBoxLayout(style_tab)
        style_layout.setContentsMargins(6, 6, 6, 6)
        style_layout.setSpacing(5)

        image_section_label = QLabel("Left image panel")
        image_section_label.setStyleSheet("font-weight: 600;")
        style_layout.addWidget(image_section_label)

        image_title_row = QHBoxLayout()
        image_title_row.addWidget(QLabel("Title:"))
        self.image_title_edit = QLineEdit()
        self.image_title_edit.setPlaceholderText("Optional image-panel title")
        self.image_title_edit.textChanged.connect(self.on_controls_changed)
        image_title_row.addWidget(self.image_title_edit, stretch=1)
        style_layout.addLayout(image_title_row)

        image_axis_row = QHBoxLayout()
        image_axis_row.addWidget(QLabel("X label:"))
        self.image_x_label_edit = QLineEdit()
        self.image_x_label_edit.setPlaceholderText("Optional")
        self.image_x_label_edit.textChanged.connect(self.on_controls_changed)
        image_axis_row.addWidget(self.image_x_label_edit, stretch=1)
        image_axis_row.addWidget(QLabel("Guide-bar label:"))
        self.image_y_label_edit = QLineEdit()
        self.image_y_label_edit.setPlaceholderText("Optional; replaces guide-bar label")
        self.image_y_label_edit.textChanged.connect(self.on_controls_changed)
        image_axis_row.addWidget(self.image_y_label_edit, stretch=1)
        style_layout.addLayout(image_axis_row)

        profile_section_label = QLabel("Right line-profile panel")
        profile_section_label.setStyleSheet("font-weight: 600;")
        style_layout.addWidget(profile_section_label)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(self.on_controls_changed)
        title_row.addWidget(self.title_edit, stretch=1)
        style_layout.addLayout(title_row)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("X label:"))
        self.x_label_edit = QLineEdit()
        self.x_label_edit.setPlaceholderText("Auto: physical distance if pixel size is available")
        self.x_label_edit.textChanged.connect(self.on_controls_changed)
        axis_row.addWidget(self.x_label_edit, stretch=1)
        axis_row.addWidget(QLabel("Left Y label:"))
        self.y_label_edit = QLineEdit()
        self.y_label_edit.textChanged.connect(self.on_controls_changed)
        axis_row.addWidget(self.y_label_edit, stretch=1)
        style_layout.addLayout(axis_row)

        y_range_row = QHBoxLayout()
        y_range_row.addWidget(QLabel("Left Y range:"))
        self.y_min_edit = QLineEdit()
        self.y_min_edit.setPlaceholderText("auto min")
        self.y_min_edit.setMaximumWidth(80)
        self.y_min_edit.textChanged.connect(self.on_controls_changed)
        y_range_row.addWidget(self.y_min_edit)
        self.y_max_edit = QLineEdit()
        self.y_max_edit.setPlaceholderText("auto max")
        self.y_max_edit.setMaximumWidth(80)
        self.y_max_edit.textChanged.connect(self.on_controls_changed)
        y_range_row.addWidget(self.y_max_edit)
        y_range_row.addSpacing(12)
        y_range_row.addWidget(QLabel("Right Y range:"))
        self.right_y_min_edit = QLineEdit()
        self.right_y_min_edit.setPlaceholderText("auto min")
        self.right_y_min_edit.setMaximumWidth(80)
        self.right_y_min_edit.textChanged.connect(self.on_controls_changed)
        y_range_row.addWidget(self.right_y_min_edit)
        self.right_y_max_edit = QLineEdit()
        self.right_y_max_edit.setPlaceholderText("auto max")
        self.right_y_max_edit.setMaximumWidth(80)
        self.right_y_max_edit.textChanged.connect(self.on_controls_changed)
        y_range_row.addWidget(self.right_y_max_edit)
        y_range_row.addStretch()
        style_layout.addLayout(y_range_row)

        legend_row = QHBoxLayout()
        legend_row.addWidget(QLabel("Legend:"))
        self.legend_position_combo = QComboBox()
        self.legend_position_combo.addItems(["Inside plot", "Outside right"])
        self.legend_position_combo.setToolTip("Move the line-profile legend outside the plot when many curves make the figure crowded.")
        self.legend_position_combo.currentTextChanged.connect(self.on_controls_changed)
        legend_row.addWidget(self.legend_position_combo)
        legend_row.addStretch()
        style_layout.addLayout(legend_row)

        style_layout.addStretch()
        self.tabs.addTab(style_tab, "Style")


        layout.addWidget(self.tabs)

        self.status_label = QLabel("Choose a background image, click two points or type coordinates, and adjust the plot settings.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        set_page_layout(self, layout)

    def make_help_button(self, title: str, text: str) -> QPushButton:
        button = QPushButton("Read instructions")
        button.setFixedWidth(130)
        button.setToolTip(title)
        button.clicked.connect(lambda _checked=False, t=title, m=text: QMessageBox.information(self, t, m))
        return button

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 9: Line-profile settings")
        else:
            self.setTitle("Step 8: Line-profile settings")

        self.state.prune_line_profile_settings_to_current_outputs()
        self.populate_controls()

        try:
            self.load_data_cache()
            self.status_label.setText(
                "Choose a background image, click two points on the map or type coordinates, and adjust the plot settings."
            )
        except Exception as e:
            self.raw_arrays_cache = {}
            self.output_arrays_cache = {}
            QMessageBox.warning(
                self,
                "Could not prepare line-profile preview",
                f"The app could not prepare the line-profile preview.\n\n{str(e)}"
            )
            self.status_label.setText("Line-profile preview is unavailable. Please go back and check earlier settings.")

        self.update_enabled_state()
        self.redraw_all()

    def get_base_output_names(self) -> List[str]:
        names = []
        for name in self.state.get_ordered_names():
            if name not in names:
                names.append(name)
        for name in self.state.get_output_names():
            if name not in names:
                names.append(name)
        return names

    def populate_controls(self):
        self._updating_controls = True
        output_names = self.state.get_output_names()
        base_output_names = self.get_base_output_names()
        lp = self.state.line_profile_settings

        if lp.base_output_name not in base_output_names:
            lp.base_output_name = base_output_names[0] if base_output_names else ""
            lp.start_x = None
            lp.start_y = None
            lp.end_x = None
            lp.end_y = None

        if not lp.selected_output_names:
            lp.selected_output_names = list(output_names)

        lp.enabled = True
        self.enable_combo.setCurrentText("Yes")

        self.base_output_combo.clear()
        self.base_output_combo.addItems(base_output_names)
        if lp.base_output_name:
            self.base_output_combo.setCurrentText(lp.base_output_name)

        self.normalize_combo.setCurrentText(lp.normalize_mode if lp.normalize_mode else "Raw values")
        self.image_title_edit.setText(lp.image_title or "")
        self.image_x_label_edit.setText(lp.image_x_label or "")
        self.image_y_label_edit.setText(lp.image_y_label or "")
        self.title_edit.setText(lp.title or "")
        x_label_text = "" if lp.x_label == "Distance along line (pixels)" else (lp.x_label or "")
        self.x_label_edit.setText(x_label_text)
        self.y_label_edit.setText(lp.y_label or "Value")
        self.y_min_edit.setText(lp.y_min or "")
        self.y_max_edit.setText(lp.y_max or "")
        self.right_y_min_edit.setText(lp.right_y_min or "")
        self.right_y_max_edit.setText(lp.right_y_max or "")
        self.legend_position_combo.setCurrentText(normalize_line_profile_legend_position(lp.legend_position))

        self.curve_table.blockSignals(True)
        self.curve_table.setRowCount(0)
        for idx, output_name in enumerate(output_names):
            row = self.curve_table.rowCount()
            self.curve_table.insertRow(row)

            output_item = QTableWidgetItem(output_name)
            output_item.setFlags(output_item.flags() & ~Qt.ItemIsEditable)
            self.curve_table.setItem(row, 0, output_item)

            plot_item = QTableWidgetItem("")
            plot_item.setFlags(plot_item.flags() | Qt.ItemIsUserCheckable)
            plot_item.setCheckState(Qt.Checked if output_name in lp.selected_output_names else Qt.Unchecked)
            self.curve_table.setItem(row, 1, plot_item)

            curve_settings = lp.curve_settings.get(output_name)
            default_color = DEFAULT_LINE_PROFILE_COLORS[idx % len(DEFAULT_LINE_PROFILE_COLORS)]
            color_item = QTableWidgetItem(curve_settings.color if curve_settings and curve_settings.color else default_color)
            self.curve_table.setItem(row, 2, color_item)

            width_item = QTableWidgetItem(curve_settings.line_width if curve_settings and curve_settings.line_width else "2.0")
            self.curve_table.setItem(row, 3, width_item)
        self.curve_table.blockSignals(False)

        self._updating_controls = False
        self.sync_controls_to_state()
        self.update_coordinate_fields_from_state()

    @staticmethod
    def format_coordinate_value(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{float(value):.6g}"

    def update_coordinate_fields_from_state(self):
        lp = self.state.line_profile_settings
        self._updating_controls = True
        self.start_x_edit.setText(self.format_coordinate_value(lp.start_x))
        self.start_y_edit.setText(self.format_coordinate_value(lp.start_y))
        self.end_x_edit.setText(self.format_coordinate_value(lp.end_x))
        self.end_y_edit.setText(self.format_coordinate_value(lp.end_y))
        self._updating_controls = False

    def parse_optional_coordinate(self, text: str, field_name: str) -> Optional[float]:
        value_text = text.strip()
        if value_text == "":
            return None
        return parse_float_user_text(value_text, field_name)

    def on_coordinate_text_changed(self, *_args):
        if self._updating_controls:
            return
        if self.enable_combo.currentText() != "Yes":
            return

        try:
            sx = self.parse_optional_coordinate(self.start_x_edit.text(), "Start x")
            sy = self.parse_optional_coordinate(self.start_y_edit.text(), "Start y")
            ex = self.parse_optional_coordinate(self.end_x_edit.text(), "End x")
            ey = self.parse_optional_coordinate(self.end_y_edit.text(), "End y")
        except ValueError:
            self.status_label.setText("Coordinate boxes contain an invalid number. Use values like 10, 10.5, or 10,5.")
            return

        lp = self.state.line_profile_settings
        if sx is not None and sy is not None:
            lp.start_x, lp.start_y = self.clamp_point_to_base_array(sx, sy)
        else:
            lp.start_x = None
            lp.start_y = None
            lp.end_x = None
            lp.end_y = None
            self.preview_point = None
            self.redraw_all()
            return

        if ex is not None and ey is not None:
            lp.end_x, lp.end_y = self.clamp_point_to_base_array(ex, ey)
            self.preview_point = None
            self.status_label.setText(
                f"Line selected: start=({lp.start_x:.1f}, {lp.start_y:.1f}), "
                f"end=({lp.end_x:.1f}, {lp.end_y:.1f})."
            )
        else:
            lp.end_x = None
            lp.end_y = None
            self.preview_point = None
            self.status_label.setText("Start point entered. Enter an end point or click the map to finish the line.")

        self.redraw_all()

    def load_data_cache(self):
        self.raw_arrays_cache = self.wizard().engine.load_raw_arrays()
        working_arrays = self.wizard().engine.apply_processing_rules(self.raw_arrays_cache)
        self.output_arrays_cache = {name: np.array(arr, copy=True) for name, arr in working_arrays.items()}
        self.output_arrays_cache.update(
            self.wizard().engine.build_output_arrays(
                self.raw_arrays_cache,
                working_arrays,
            )
        )

    def get_curve_table_settings(self) -> Tuple[List[str], Dict[str, LineProfileCurveSettings]]:
        selected = []
        curve_settings: Dict[str, LineProfileCurveSettings] = {}

        for row in range(self.curve_table.rowCount()):
            output_item = self.curve_table.item(row, 0)
            plot_item = self.curve_table.item(row, 1)
            color_item = self.curve_table.item(row, 2)
            width_item = self.curve_table.item(row, 3)

            if output_item is None:
                continue

            output_name = output_item.text().strip()
            enabled = plot_item is not None and plot_item.checkState() == Qt.Checked
            if enabled:
                selected.append(output_name)

            curve_settings[output_name] = LineProfileCurveSettings(
                enabled=enabled,
                color=color_item.text().strip() if color_item else "",
                line_width=width_item.text().strip() if width_item else "2.0",
            )

        return selected, curve_settings

    def sync_controls_to_state(self):
        if self._updating_controls:
            return

        lp = self.state.line_profile_settings
        lp.enabled = self.enable_combo.currentText() == "Yes"
        lp.base_output_name = self.base_output_combo.currentText().strip()
        lp.normalize_mode = self.normalize_combo.currentText().strip() or "Raw values"
        lp.image_title = self.image_title_edit.text().strip()
        lp.image_x_label = self.image_x_label_edit.text().strip()
        lp.image_y_label = self.image_y_label_edit.text().strip()
        lp.title = self.title_edit.text().strip()
        lp.x_label = self.x_label_edit.text().strip()
        lp.y_label = self.y_label_edit.text().strip() or "Value"
        lp.y_min = self.y_min_edit.text().strip()
        lp.y_max = self.y_max_edit.text().strip()
        lp.right_y_min = self.right_y_min_edit.text().strip()
        lp.right_y_max = self.right_y_max_edit.text().strip()
        lp.legend_position = normalize_line_profile_legend_position(self.legend_position_combo.currentText())

        selected, curve_settings = self.get_curve_table_settings()
        lp.selected_output_names = selected
        lp.curve_settings = curve_settings

    def update_enabled_state(self):
        enabled = self.enable_combo.currentText() == "Yes"
        widgets = [
            self.base_output_combo,
            self.normalize_combo,
            self.image_title_edit,
            self.image_x_label_edit,
            self.image_y_label_edit,
            self.title_edit,
            self.x_label_edit,
            self.y_label_edit,
            self.y_min_edit,
            self.y_max_edit,
            self.right_y_min_edit,
            self.right_y_max_edit,
            self.legend_position_combo,
            self.combined_canvas,
            self.clear_line_button,
            self.curve_table,
            self.start_x_edit,
            self.start_y_edit,
            self.end_x_edit,
            self.end_y_edit,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def on_controls_changed(self, *_args):
        if self._updating_controls:
            return
        self.sync_controls_to_state()
        self.update_enabled_state()
        self.redraw_all()

    def on_base_output_changed(self, *_args):
        if self._updating_controls:
            return
        lp = self.state.line_profile_settings
        old_base = lp.base_output_name
        self.sync_controls_to_state()
        if old_base != lp.base_output_name:
            lp.start_x = None
            lp.start_y = None
            lp.end_x = None
            lp.end_y = None
            self.preview_point = None
            self.update_coordinate_fields_from_state()
        self.redraw_all()

    def on_curve_table_changed(self, _item):
        if self._updating_controls:
            return
        self.sync_controls_to_state()
        self.redraw_all()

    def clear_line(self):
        lp = self.state.line_profile_settings
        lp.start_x = None
        lp.start_y = None
        lp.end_x = None
        lp.end_y = None
        self.preview_point = None
        self.update_coordinate_fields_from_state()
        self.status_label.setText("Line cleared. Click the start point, then the end point, or type coordinates manually.")
        self.redraw_all()

    def clamp_point_to_base_array(self, x: float, y: float) -> Tuple[float, float]:
        base_name = self.base_output_combo.currentText().strip()
        if base_name not in self.output_arrays_cache:
            return x, y
        rows, cols = self.output_arrays_cache[base_name].shape
        return (
            float(np.clip(x, 0, cols - 1)),
            float(np.clip(y, 0, rows - 1)),
        )

    def on_map_click(self, event):
        if self.enable_combo.currentText() != "Yes":
            return
        if event.inaxes != self.map_ax or event.xdata is None or event.ydata is None:
            return

        x, y = self.clamp_point_to_base_array(event.xdata, event.ydata)
        lp = self.state.line_profile_settings

        if lp.start_x is None or lp.start_y is None or (lp.end_x is not None and lp.end_y is not None):
            lp.start_x = x
            lp.start_y = y
            lp.end_x = None
            lp.end_y = None
            self.preview_point = None
            self.status_label.setText("Start point selected. Move the mouse if you want a preview, then click the end point.")
        else:
            lp.end_x = x
            lp.end_y = y
            self.preview_point = None
            self.status_label.setText(
                f"Line selected: start=({lp.start_x:.1f}, {lp.start_y:.1f}), "
                f"end=({lp.end_x:.1f}, {lp.end_y:.1f})."
            )

        self.update_coordinate_fields_from_state()
        self.redraw_all()

    def on_map_motion(self, event):
        lp = self.state.line_profile_settings
        if self.enable_combo.currentText() != "Yes":
            return
        if lp.start_x is None or lp.start_y is None or lp.end_x is not None or lp.end_y is not None:
            return
        if event.inaxes != self.map_ax or event.xdata is None or event.ydata is None:
            return

        self.preview_point = self.clamp_point_to_base_array(event.xdata, event.ydata)
        self.redraw_all()

    def get_display_settings_for_base(self, base_name: str) -> OutputDisplaySettings:
        return self.state.display_settings.get(
            base_name,
            OutputDisplaySettings(
                mode="Continuous",
                colormap="gray" if is_grayscale_name(base_name) else "viridis",
                colorbar_label=default_colorbar_label(base_name),
            ),
        )

    def current_line_coords(self) -> Optional[Tuple[float, float, float, float]]:
        lp = self.state.line_profile_settings
        if lp.start_x is None or lp.start_y is None:
            return None
        if lp.end_x is not None and lp.end_y is not None:
            return (lp.start_x, lp.start_y, lp.end_x, lp.end_y)
        if self.preview_point is not None:
            return (lp.start_x, lp.start_y, self.preview_point[0], self.preview_point[1])
        return None

    def draw_combined_preview(self):
        self.combined_figure.clear()

        if self.enable_combo.currentText() != "Yes":
            self.map_ax = None
            self.profile_ax = None
            self.combined_canvas.draw_idle()
            return

        self.sync_controls_to_state()
        lp = self.state.line_profile_settings
        legend_outside = normalize_line_profile_legend_position(lp.legend_position) == "Outside right"

        # Keep the live preview close to the exported two-panel layout. The main
        # map and profile axes use the same requested panel size; colorbar and
        # labels live outside the main image panel. If the legend is outside,
        # leave a little more room on the right side of the live canvas.
        panel_y = 0.18
        panel_h = 0.66
        panel_w = 0.29 if legend_outside else 0.32
        map_x = 0.15
        profile_x = 0.52 if legend_outside else 0.55
        self.map_ax = self.combined_figure.add_axes([map_x, panel_y, panel_w, panel_h])
        self.profile_ax = self.combined_figure.add_axes([profile_x, panel_y, panel_w, panel_h])
        cax = self.combined_figure.add_axes([map_x - 0.022, panel_y, 0.012, panel_h])

        base_name = self.base_output_combo.currentText().strip()

        if base_name == "" or base_name not in self.output_arrays_cache:
            self.map_ax.text(0.5, 0.5, "No output available", ha="center", va="center")
            self.map_ax.set_axis_off()
            self.profile_ax.set_axis_off()
            self.combined_canvas.draw_idle()
            return

        arr = self.output_arrays_cache[base_name]
        display_settings = self.get_display_settings_for_base(base_name)
        mask_layers = self.wizard().engine.build_grayscale_mask_layers(self.raw_arrays_cache, output_name=base_name)
        pixel_size_um = None
        try:
            pixel_size_um = self.wizard().engine.get_pixel_size_um_for_output(base_name)
        except Exception:
            pixel_size_um = None

        line_coords = self.current_line_coords()
        try:
            self.wizard().engine.plot_output_map_axis(
                self.map_ax,
                arr,
                base_name,
                display_settings,
                mask_layers=mask_layers,
                line_coords=line_coords,
                show_colorbar=True,
                add_scale_bar=False,
                pixel_size_um=pixel_size_um,
                colorbar_axis=cax,
                colorbar_side_override="left",
                colorbar_label_override=lp.image_y_label.strip() or None,
                show_line_length_label=True,
            )
        except Exception as e:
            self.map_ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
            self.map_ax.set_axis_off()

        if self.map_ax is not None and self.map_ax.has_data():
            if lp.image_title.strip():
                self.map_ax.set_title(lp.image_title.strip(), fontname=DEFAULT_FONT_NAME)
            if lp.image_x_label.strip():
                self.map_ax.set_xlabel(lp.image_x_label.strip(), fontname=DEFAULT_FONT_NAME)

        # If only the start point is known, show it even though there is no full line yet.
        if lp.start_x is not None and lp.start_y is not None and line_coords is None:
            self.map_ax.plot(lp.start_x, lp.start_y, marker="o", color="white", markeredgecolor="black")

        if not lp.selected_output_names:
            self.profile_ax.text(0.5, 0.5, "Select at least one output to plot", ha="center", va="center")
            self.profile_ax.set_axis_off()
        elif lp.start_x is None or lp.start_y is None or lp.end_x is None or lp.end_y is None:
            self.profile_ax.text(0.5, 0.5, "Click two points on the map", ha="center", va="center")
            self.profile_ax.set_axis_off()
        else:
            try:
                profile_data = self.wizard().engine.extract_line_profile_data(
                    self.output_arrays_cache,
                    lp.base_output_name,
                    lp.selected_output_names,
                    lp.start_x,
                    lp.start_y,
                    lp.end_x,
                    lp.end_y,
                    lp.normalize_mode,
                )
                self.wizard().engine.plot_line_profile_axis(
                    self.profile_ax,
                    profile_data,
                    title_override=lp.title.strip(),
                )
            except Exception as e:
                self.profile_ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
                self.profile_ax.set_axis_off()

        try:
            self.combined_figure.canvas.draw()
            map_pos = self.map_ax.get_position()
            profile_pos = self.profile_ax.get_position()
            cax.set_position([map_pos.x0 - 0.022, map_pos.y0, 0.012, map_pos.height])
            new_profile_position = [profile_pos.x0, map_pos.y0, map_pos.width, map_pos.height]
            self.profile_ax.set_position(new_profile_position)
            right_profile_ax = getattr(self.profile_ax, "_elementti_right_axis", None)
            if right_profile_ax is not None:
                right_profile_ax.set_position(new_profile_position)
        except Exception:
            pass

        self.combined_canvas.draw_idle()

    def redraw_all(self):
        self.draw_combined_preview()

    def validate_y_axis_limits(self) -> bool:
        for axis_label, min_text, max_text in [
            ("Left y-axis", self.y_min_edit.text().strip(), self.y_max_edit.text().strip()),
            ("Right y-axis", self.right_y_min_edit.text().strip(), self.right_y_max_edit.text().strip()),
        ]:
            min_value = None
            max_value = None
            if min_text:
                try:
                    min_value = parse_float_user_text(min_text, f"{axis_label} minimum")
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid y-axis limit", str(exc))
                    return False
            if max_text:
                try:
                    max_value = parse_float_user_text(max_text, f"{axis_label} maximum")
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid y-axis limit", str(exc))
                    return False
            if min_value is not None and max_value is not None and min_value >= max_value:
                QMessageBox.warning(
                    self,
                    "Invalid y-axis limit",
                    f"{axis_label} minimum must be smaller than {axis_label} maximum.",
                )
                return False
        return True


    def validate_curve_settings(self) -> bool:
        for row in range(self.curve_table.rowCount()):
            output_item = self.curve_table.item(row, 0)
            color_item = self.curve_table.item(row, 2)
            width_item = self.curve_table.item(row, 3)
            output_name = output_item.text() if output_item else f"row {row + 1}"

            color = color_item.text().strip() if color_item else ""
            if color and not is_color_like(color):
                QMessageBox.warning(
                    self,
                    "Invalid line color",
                    f"'{color}' is not a valid line color for '{output_name}'.\n\n"
                    "Use names like red, blue, orange, green, black, or hex codes like #FF0000."
                )
                return False

            width_text = width_item.text().strip() if width_item else ""
            if width_text:
                try:
                    width = parse_float_user_text(width_text, "Line width")
                    if width <= 0:
                        raise ValueError
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid line width",
                        f"Line width for '{output_name}' must be a number greater than 0."
                    )
                    return False
        return True

    def validatePage(self):
        self.sync_controls_to_state()
        lp = self.state.line_profile_settings

        if not lp.enabled:
            return True

        if not self.output_arrays_cache:
            try:
                self.load_data_cache()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Line-profile data unavailable",
                    f"The app could not prepare the line-profile data.\n\n{str(e)}"
                )
                return False

        if not lp.base_output_name:
            QMessageBox.warning(self, "Missing background output", "Please choose a background output for the line profile.")
            return False

        if not lp.selected_output_names:
            QMessageBox.warning(self, "No profile curves", "Please select at least one output to plot in the line profile.")
            return False

        if lp.start_x is None or lp.start_y is None or lp.end_x is None or lp.end_y is None:
            QMessageBox.warning(self, "No complete line", "Please click two points on the map, or choose 'No' for line profile.")
            return False

        if not self.validate_curve_settings():
            return False

        if not self.validate_y_axis_limits():
            return False

        try:
            self.wizard().engine.extract_line_profile_data(
                self.output_arrays_cache,
                lp.base_output_name,
                lp.selected_output_names,
                lp.start_x,
                lp.start_y,
                lp.end_x,
                lp.end_y,
                lp.normalize_mode,
            )
        except Exception as e:
            QMessageBox.warning(self, "Invalid line profile", str(e))
            return False

        return True


class GeneratePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 9: Choose output folder and generate")
        self.setSubTitle("Save outputs.")

        layout = QVBoxLayout()
        set_compact_layout(layout)

        add_compact_help_row(
            layout,
            self,
            "Choose export folder, file options, image format, and DPI.",
            "Generate/export help",
            "Use this page to choose where Elementti will save the results from the current run. Select an output folder and optionally enter a project name. If a project name is provided, it is added to the beginning of exported file names.\n\n"
            "You can save figure files, processed CSV files, or both. Figure files use the selected format and DPI. Processed CSV files contain the numerical arrays after scaling and active numerical processing rules. Ratio and formula CSV files contain the calculated values, with non-finite values stored as NaN.\n\n"
            "Elementti also saves a structured JSON summary file and a plain-text methods file. These files record the selected input names, preprocessing settings, masking/noise settings, display settings, output settings, and line-profile settings if used.\n\n"
            "After clicking Generate, the app saves the outputs and remains open, so you can go back, change settings, and generate another set of outputs."
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Output folder:"))
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Choose a folder")
        self.output_folder_edit.textChanged.connect(self.on_output_folder_changed)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_output_folder)
        row.addWidget(self.output_folder_edit)
        row.addWidget(self.browse_button)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Project name (optional):"))
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Example: corrosion_run_01")
        self.project_name_edit.textChanged.connect(self.on_project_name_changed)
        row.addWidget(self.project_name_edit)
        layout.addLayout(row)

        self.save_png_checkbox = QCheckBox("Save figure files")
        self.save_png_checkbox.setChecked(True)
        self.save_png_checkbox.stateChanged.connect(self.on_save_options_changed)

        self.save_csv_checkbox = QCheckBox("Save processed CSV files")
        self.save_csv_checkbox.setChecked(True)
        self.save_csv_checkbox.stateChanged.connect(self.on_save_options_changed)

        layout.addWidget(self.save_png_checkbox)
        layout.addWidget(self.save_csv_checkbox)

        row = QHBoxLayout()
        row.addWidget(QLabel("Figure format:"))
        self.figure_format_combo = QComboBox()
        self.figure_format_combo.addItems(["png", "jpg", "jpeg", "tif", "tiff"])
        self.figure_format_combo.currentTextChanged.connect(self.on_figure_options_changed)
        row.addWidget(self.figure_format_combo)

        row.addWidget(QLabel("DPI:"))
        self.dpi_edit = QLineEdit()
        self.dpi_edit.setPlaceholderText("300")
        self.dpi_edit.textChanged.connect(self.on_figure_options_changed)
        row.addWidget(self.dpi_edit)
        row.addStretch()
        layout.addLayout(row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.citation_title_label = QLabel(f"Citation for {APP_NAME} {APP_VERSION}:")
        layout.addWidget(self.citation_title_label)

        self.citation_label = QLabel(APP_CITATION_TEXT)
        self.citation_label.setWordWrap(True)
        self.citation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.citation_label)

        set_page_layout(self, layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 9: Choose output folder and generate")
        else:
            self.setTitle("Step 8: Choose output folder and generate")

        self.output_folder_edit.setText(self.state.output_folder)
        self.project_name_edit.setText(self.state.project_name)
        self.save_png_checkbox.setChecked(self.state.save_png)
        self.save_csv_checkbox.setChecked(self.state.save_csv)
        self.figure_format_combo.setCurrentText(self.state.figure_format)
        self.dpi_edit.setText(str(self.state.figure_dpi))
        self.sync_to_state()
        self.update_summary()

    def isComplete(self):
        return self.output_folder_edit.text().strip() != ""

    def sync_to_state(self):
        self.state.output_folder = self.output_folder_edit.text().strip()
        self.state.project_name = self.project_name_edit.text().strip()
        self.state.save_png = self.save_png_checkbox.isChecked()
        self.state.save_csv = self.save_csv_checkbox.isChecked()
        self.state.figure_format = self.figure_format_combo.currentText().strip()

        dpi_text = self.dpi_edit.text().strip()
        if dpi_text:
            try:
                self.state.figure_dpi = int(dpi_text)
            except ValueError:
                pass

    def on_output_folder_changed(self, _text):
        self.sync_to_state()
        self.update_summary()
        self.completeChanged.emit()

    def on_project_name_changed(self, _text):
        self.sync_to_state()
        self.update_summary()

    def on_save_options_changed(self, _state):
        self.sync_to_state()
        self.update_summary()

    def on_figure_options_changed(self, *_args):
        self.sync_to_state()
        self.update_summary()

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            self.output_folder_edit.text().strip() or "",
        )
        if folder:
            self.output_folder_edit.setText(folder)
            self.sync_to_state()
            self.update_summary()
            self.completeChanged.emit()

    def validatePage(self):
        self.sync_to_state()

        if self.state.output_folder == "":
            QMessageBox.warning(self, "Missing output folder", "Please choose an output folder.")
            return False

        if not self.state.save_png and not self.state.save_csv:
            QMessageBox.warning(
                self,
                "Nothing to save",
                "Please select at least one output type: figure files and/or processed CSV files."
            )
            return False

        if self.state.save_png:
            dpi_text = self.dpi_edit.text().strip()
            if dpi_text == "":
                QMessageBox.warning(self, "Missing DPI", "Please enter a DPI value.")
                return False

            try:
                dpi_value = int(dpi_text)
                if dpi_value <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid DPI", "DPI must be a whole number greater than 0.")
                return False

            self.state.figure_dpi = dpi_value

        try:
            os.makedirs(self.state.output_folder, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Cannot create output folder",
                f"The app could not create or access the output folder.\n\n{str(e)}"
            )
            return False

        return True

    def update_summary(self):
        folder = self.output_folder_edit.text().strip()
        fig_text = "Yes" if self.save_png_checkbox.isChecked() else "No"
        csv_text = "Yes" if self.save_csv_checkbox.isChecked() else "No"
        fmt = self.figure_format_combo.currentText().strip()
        dpi = self.dpi_edit.text().strip() or "(not set yet)"
        masking_text = "Yes" if self.state.masking_and_noise_settings.enabled else "No"
        line_profile_text = "Yes" if self.state.line_profile_settings.enabled else "No"

        self.summary_label.setText(
            f"Summary:\n"
            f"- Selected maps: {len(self.state.selected_maps)}\n"
            f"- Selected ratios: {len(self.state.selected_ratios)}\n"
            f"- Selected formulas: {len(self.state.selected_formulas)}\n"
            f"- Masking and noise removal enabled: {masking_text}\n"
            f"- Line profile enabled: {line_profile_text}\n"
            f"- Save figure files: {fig_text}\n"
            f"- Save processed CSV files: {csv_text}\n"
            f"- Figure format: {fmt}\n"
            f"- DPI: {dpi}\n"
            f"- Output folder: {folder if folder else '(not chosen yet)'}"
        )


class ElementtiWizard(QWizard):
    PAGE_WELCOME = 0
    PAGE_UPLOAD = 1
    PAGE_RENAME = 2
    PAGE_OUTPUTS = 3
    PAGE_PROCESSING = 4
    PAGE_MASKING_INTRO = 5
    PAGE_MASKING_SETTINGS = 6
    PAGE_DISPLAY = 7
    PAGE_LINE_PROFILE_INTRO = 8
    PAGE_LINE_PROFILE = 9
    PAGE_GENERATE = 10

    def __init__(self):
        super().__init__()

        self.state = AppState()
        self.engine = ProcessingEngine(self.state)

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setWizardStyle(QWizard.ModernStyle)

        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.setPage(self.PAGE_WELCOME, WelcomePage())
        self.setPage(self.PAGE_UPLOAD, UploadPage())
        self.setPage(self.PAGE_RENAME, RenamePage())
        self.setPage(self.PAGE_OUTPUTS, OutputsPage())
        self.setPage(self.PAGE_PROCESSING, ProcessingSettingsPage())
        self.setPage(self.PAGE_MASKING_INTRO, MaskingIntroPage())
        self.setPage(self.PAGE_MASKING_SETTINGS, MaskingSettingsPage())
        self.setPage(self.PAGE_DISPLAY, DisplaySettingsPage())
        self.setPage(self.PAGE_LINE_PROFILE_INTRO, LineProfileIntroPage())
        self.setPage(self.PAGE_LINE_PROFILE, LineProfilePage())
        self.setPage(self.PAGE_GENERATE, GeneratePage())

        self.setStartId(self.PAGE_WELCOME)

        if self.layout() is not None:
            self.layout().setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        self.setMinimumSize(820, 560)
        self.setMaximumSize(16777215, 16777215)
        self.resize(1000, 650)

        self.setButtonText(QWizard.FinishButton, "Generate")
        self.setButtonText(QWizard.CancelButton, "Close")

    def cleanup_demo_data(self):
        cleanup_demo_sample_folder(self.state.sample_data_folder)
        self.state.sample_data_folder = ""
        self.state.sample_data_seed = None
        if self.state.sample_data_mode == "demo":
            self.state.sample_data_mode = ""

    def reject(self):
        self.cleanup_demo_data()
        super().reject()

    def closeEvent(self, event):
        self.cleanup_demo_data()
        super().closeEvent(event)

    def accept(self):
        try:
            summary_path, methods_path = self.engine.generate_outputs()

            msg_lines = [
                "The outputs were generated successfully.",
                "",
                "Summary file:",
                summary_path,
                "",
                "Methods file:",
                methods_path,
                "",
                "The app will stay open so you can go back and revise your settings.",
            ]

            if self.state.save_png:
                msg_lines.append(f"Figure files were saved as .{self.state.figure_format}.")
            if self.state.save_csv:
                msg_lines.append("Processed CSV files were saved.")
            if self.state.masking_and_noise_settings.enabled:
                msg_lines.append("Masking and noise-removal settings were applied.")
            if self.state.line_profile_settings.enabled:
                msg_lines.append("Line-profile outputs were saved according to the selected save options.")

            msg = QMessageBox(self)
            msg.setWindowTitle("Finished")
            msg.setIcon(QMessageBox.Information)
            msg.setText("\n".join(msg_lines))

            open_folder_button = msg.addButton("Open output folder", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Close)

            msg.exec()

            if msg.clickedButton() == open_folder_button:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.state.output_folder)))

        except Exception as e:
            QMessageBox.warning(
                self,
                "Generation failed",
                f"The app could not generate the outputs.\n\n{str(e)}"
            )


def set_windows_app_user_model_id() -> None:
    """Help Windows show Elementti's own icon instead of the generic Python icon."""
    if sys.platform != "win32":
        return
    try:
        app_id = f"AboAkademi.Elementti.{APP_VERSION}"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def main():
    set_windows_app_user_model_id()
    app = QApplication(sys.argv)

    icon_path = find_app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    wizard = ElementtiWizard()
    if icon_path:
        wizard.setWindowIcon(QIcon(icon_path))
    wizard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

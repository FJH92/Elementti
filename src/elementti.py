import sys
import os
import io
import csv
import json
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from PySide6.QtGui import QDesktopServices
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
)


APP_DEVELOPER_TEXT = (
    "Elementti was developed by the Laboratory of High Temperature Processes and Materials "
    "at Åbo Akademi University, Finland."
)

APP_CITATION_TEXT = (
    "If you use Elementti in your work, please cite: "
    "F. Jafarihonar and E. Vainio, 2026. "
    "Elementti: A Python-based GUI for processing and visualization of SEM-EDS elemental maps."
)


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


def parse_bins_text(bins_text: str) -> List[float]:
    parts = [p.strip() for p in bins_text.split(",") if p.strip()]
    return [float(p) for p in parts]


def parse_color_list(color_text: str) -> List[str]:
    return [c.strip() for c in color_text.split(",") if c.strip()]


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
    cleaned = text.strip()
    if cleaned == "":
        return []

    items = [x.strip() for x in cleaned.split(",") if x.strip()]
    result = []

    for item in items:
        try:
            value = int(item)
        except ValueError:
            raise ValueError(f"{what} must be comma-separated whole numbers like 1,2,3")

        if value <= 0:
            raise ValueError(f"{what} must use 1-based positive numbers like 1,2,3")

        result.append(value - 1)

    return sorted(set(result))


def create_sample_data_files(folder: str) -> List[str]:
    rows = 384
    cols = 512

    y = np.linspace(-1.0, 1.0, rows)
    x = np.linspace(-1.0, 1.0, cols)
    X, Y = np.meshgrid(x, y)

    rng = np.random.default_rng()

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

    grayscale_path = os.path.join(folder, "grayscale_demo.csv")
    element1_path = os.path.join(folder, "element1_demo.csv")
    element2_path = os.path.join(folder, "element2_demo.csv")

    np.savetxt(grayscale_path, grayscale, delimiter=",", fmt="%.6f")
    np.savetxt(element1_path, element1, delimiter=",", fmt="%.6f")
    np.savetxt(element2_path, element2, delimiter=",", fmt="%.6f")

    return [grayscale_path, element1_path, element2_path]


@dataclass
class FileReadSettings:
    manual_rows: str = ""
    manual_columns: str = ""
    scale_factor: str = "0.01"


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
class LowValueRule:
    enabled: bool = False
    floor: str = ""


@dataclass
class GrayscaleMaskRule:
    enabled: bool = True
    source_name: str = ""
    condition: str = "below"   # below, above, between, outside
    value_a: str = ""
    value_b: str = ""
    color: str = "white"


@dataclass
class ConditionalReplacementRule:
    enabled: bool = True
    condition_source_name: str = ""
    condition_operator: str = ">"
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
class AppState:
    selected_file_paths: List[str] = field(default_factory=list)
    renamed_names_by_path: Dict[str, str] = field(default_factory=dict)
    selected_maps: List[str] = field(default_factory=list)
    selected_ratios: List[Tuple[str, str]] = field(default_factory=list)
    file_processing_settings: Dict[str, FileReadSettings] = field(default_factory=dict)
    processed_min_max_by_name: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    display_settings: Dict[str, OutputDisplaySettings] = field(default_factory=dict)
    masking_and_noise_settings: MaskingAndNoiseSettings = field(default_factory=MaskingAndNoiseSettings)
    output_folder: str = ""
    project_name: str = ""
    save_png: bool = True
    save_csv: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300
    sample_data_folder: str = ""

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
        arr = arr * float(settings.scale_factor)
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

            floor_value = float(floor_text)
            finite_mask = np.isfinite(arr)
            arr[finite_mask & (arr < floor_value)] = floor_value

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
            operator = rule.condition_operator.strip()
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

            replacement_value = float(replacement_text)

            if operator == "between":
                a_text = rule.value_a.strip()
                b_text = rule.value_b.strip()
                if a_text == "" or b_text == "":
                    raise ValueError(f"Conditional replacement rule {idx} requires both Value A and Value B.")
                a = float(a_text)
                b = float(b_text)
                low = min(a, b)
                high = max(a, b)
                condition_mask = np.isfinite(source_arr) & (source_arr >= low) & (source_arr <= high)
            else:
                a_text = rule.value_a.strip()
                if a_text == "":
                    raise ValueError(f"Conditional replacement rule {idx} requires Value A.")
                a = float(a_text)

                if operator == ">":
                    condition_mask = np.isfinite(source_arr) & (source_arr > a)
                elif operator == ">=":
                    condition_mask = np.isfinite(source_arr) & (source_arr >= a)
                elif operator == "<":
                    condition_mask = np.isfinite(source_arr) & (source_arr < a)
                elif operator == "<=":
                    condition_mask = np.isfinite(source_arr) & (source_arr <= a)
                else:
                    raise ValueError(f"Conditional replacement rule {idx} uses unknown operator '{operator}'.")

            target_arr[condition_mask] = replacement_value

        return output

    def apply_processing_rules(self, raw_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        working = self.apply_low_value_protection(raw_arrays)
        working = self.apply_conditional_replacements(working)
        return working

    def build_grayscale_rule_mask(
        self,
        raw_arrays_by_name: Dict[str, np.ndarray],
        rule: GrayscaleMaskRule,
        rule_index: int,
    ) -> np.ndarray:
        source_name = rule.source_name.strip()
        condition = rule.condition.strip().lower()

        if source_name == "":
            raise ValueError(f"Grayscale mask rule {rule_index} has no source file.")
        if source_name not in raw_arrays_by_name:
            raise ValueError(f"Grayscale mask rule {rule_index} uses unknown source file '{source_name}'.")

        source_array = raw_arrays_by_name[source_name]

        if condition in ("below", "above"):
            if rule.value_a.strip() == "":
                raise ValueError(f"Grayscale mask rule {rule_index} requires Value A.")
            a = float(rule.value_a)

            if condition == "below":
                return np.isfinite(source_array) & (source_array < a)
            return np.isfinite(source_array) & (source_array > a)

        if condition in ("between", "outside"):
            if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                raise ValueError(f"Grayscale mask rule {rule_index} requires both Value A and Value B.")
            a = float(rule.value_a)
            b = float(rule.value_b)
            low = min(a, b)
            high = max(a, b)

            if condition == "between":
                return np.isfinite(source_array) & (source_array >= low) & (source_array <= high)

            return np.isfinite(source_array) & ((source_array < low) | (source_array > high))

        raise ValueError(f"Grayscale mask rule {rule_index} uses unknown condition '{condition}'.")

    def build_combined_grayscale_mask(self, raw_arrays_by_name: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        settings = self.state.masking_and_noise_settings

        if not settings.enabled or not settings.grayscale_enabled:
            return None

        active_rules = [rule for rule in settings.grayscale_rules if rule.enabled]
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

    def build_grayscale_mask_layers(self, raw_arrays_by_name: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, str]]:
        settings = self.state.masking_and_noise_settings

        if not settings.enabled or not settings.grayscale_enabled:
            return []

        layers = []
        active_rules = [rule for rule in settings.grayscale_rules if rule.enabled]

        for idx, rule in enumerate(active_rules, start=1):
            rule_mask = self.build_grayscale_rule_mask(raw_arrays_by_name, rule, idx)
            layers.append((rule_mask, rule.color.strip() or "white"))

        return layers

    @staticmethod
    def get_visible_finite_values(array: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is not None and mask.shape != array.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {array.shape}."
            )

        visible = array if mask is None else array[~mask]
        return visible[np.isfinite(visible)]

    def compute_output_stats(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        raw_arrays = self.load_raw_arrays()
        working_arrays = self.apply_processing_rules(raw_arrays)
        display_mask = self.build_combined_grayscale_mask(raw_arrays)

        result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for name, arr in working_arrays.items():
            finite = self.get_visible_finite_values(arr, mask=display_mask)
            result[name] = (
                None if finite.size == 0 else float(np.min(finite)),
                None if finite.size == 0 else float(np.max(finite)),
            )

        for num, den in self.state.selected_ratios:
            if num not in working_arrays or den not in working_arrays:
                continue

            num_arr = working_arrays[num]
            den_arr = working_arrays[den]

            if num_arr.shape != den_arr.shape:
                raise ValueError(
                    f"Ratio '{num} / {den}' cannot be calculated because the two arrays "
                    f"have different shapes: {num_arr.shape} and {den_arr.shape}."
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_arr = num_arr / den_arr
                ratio_arr[~np.isfinite(ratio_arr)] = np.nan

            finite = self.get_visible_finite_values(ratio_arr, mask=display_mask)
            result[f"{num} / {den}"] = (
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

        vmin = float(settings.display_min) if settings.display_min.strip() else actual_min
        vmax = float(settings.display_max) if settings.display_max.strip() else actual_max

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

        side = "left" if settings.colorbar_side.strip().lower() == "left" else "right"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(side, size="4%", pad=0.05)

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
            cbar = plt.colorbar(im, cax=cax)

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
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_ticks(boundaries)

        else:
            plt.close(fig)
            raise ValueError(f"Unknown mode '{settings.mode}' for '{output_name}'.")

        for rule_mask, rule_color in (mask_layers or []):
            overlay = np.zeros(rule_mask.shape + (4,), dtype=float)
            overlay[rule_mask] = to_rgba(rule_color)
            ax.imshow(overlay, interpolation="nearest")

        cbar.set_label(settings.colorbar_label.strip() or output_name)

        if side == "left":
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.yaxis.tick_left()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

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
                "condition": rule.condition,
                "value_a": rule.value_a,
                "value_b": rule.value_b,
                "color": rule.color,
            })

        conditional_rules = []
        for rule in self.state.masking_and_noise_settings.conditional_rules:
            conditional_rules.append({
                "enabled": rule.enabled,
                "condition_source_name": rule.condition_source_name,
                "condition_operator": rule.condition_operator,
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

        return {
            "developer": APP_DEVELOPER_TEXT,
            "citation": APP_CITATION_TEXT,
            "renamed_files": renamed_files,
            "selected_maps": self.state.selected_maps,
            "selected_ratios": [
                {"numerator": num, "denominator": den}
                for num, den in self.state.selected_ratios
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

        lines.append("Elementti methods summary")
        lines.append("")
        lines.append(APP_DEVELOPER_TEXT)
        lines.append("")
        lines.append("Citation:")
        lines.append(APP_CITATION_TEXT)
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

        lines.append("File processing settings:")
        for path in self.state.selected_file_paths:
            name = self.state.renamed_names_by_path.get(path, "")
            settings = self.state.file_processing_settings.get(path, FileReadSettings())
            lines.append(f"- {name}")
            lines.append(f"  Rows removed: {settings.manual_rows if settings.manual_rows else 'None'}")
            lines.append(f"  Columns removed: {settings.manual_columns if settings.manual_columns else 'None'}")
            lines.append(f"  Scale factor: {settings.scale_factor}")
        lines.append("")

        ms = self.state.masking_and_noise_settings
        lines.append("Masking and noise-removal settings:")
        lines.append(f"- Enabled: {'Yes' if ms.enabled else 'No'}")

        lines.append(f"- Grayscale masking enabled: {'Yes' if ms.grayscale_enabled else 'No'}")
        if ms.grayscale_enabled:
            active_grayscale_rules = [rule for rule in ms.grayscale_rules if rule.enabled]
            if active_grayscale_rules:
                for idx, rule in enumerate(active_grayscale_rules, start=1):
                    if rule.condition == "between":
                        condition_text = f"{rule.source_name} between {rule.value_a} and {rule.value_b}"
                    elif rule.condition == "outside":
                        condition_text = f"{rule.source_name} outside {rule.value_a} and {rule.value_b}"
                    elif rule.condition == "above":
                        condition_text = f"{rule.source_name} above {rule.value_a}"
                    else:
                        condition_text = f"{rule.source_name} below {rule.value_a}"

                    lines.append(f"  - Rule {idx}: mask pixels where {condition_text}")
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
                    if rule.condition_operator == "between":
                        condition_text = f"{rule.condition_source_name} between {rule.value_a} and {rule.value_b}"
                    else:
                        condition_text = f"{rule.condition_source_name} {rule.condition_operator} {rule.value_a}"
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

        lines.append("Output options:")
        lines.append(f"- Output folder: {self.state.output_folder}")
        lines.append(f"- Project name: {self.state.project_name if self.state.project_name else 'None'}")
        lines.append(f"- Save figure files: {'Yes' if self.state.save_png else 'No'}")
        lines.append(f"- Save processed CSV files: {'Yes' if self.state.save_csv else 'No'}")
        lines.append(f"- Figure format: {self.state.figure_format}")
        lines.append(f"- Figure DPI: {self.state.figure_dpi}")

        return "\n".join(lines)

    def generate_outputs(self) -> Tuple[str, str]:
        output_folder = self.state.output_folder.strip()
        if output_folder == "":
            raise ValueError("Output folder is empty.")

        os.makedirs(output_folder, exist_ok=True)

        prefix = sanitize_filename(self.state.project_name.strip()) if self.state.project_name.strip() else ""
        ext = self.state.figure_format.strip().lower()
        dpi = int(self.state.figure_dpi)

        raw_arrays = self.load_raw_arrays()
        working_arrays = self.apply_processing_rules(raw_arrays)
        mask_layers = self.build_grayscale_mask_layers(raw_arrays)

        for map_name in self.state.selected_maps:
            if map_name not in working_arrays:
                raise ValueError(f"Selected map '{map_name}' was not loaded.")

            arr = working_arrays[map_name]
            settings = self.state.display_settings[map_name]

            file_stub = sanitize_filename(map_name)
            if prefix:
                file_stub = f"{prefix}_{file_stub}"

            if self.state.save_csv:
                csv_path = os.path.join(output_folder, f"{file_stub}.csv")
                np.savetxt(csv_path, arr, delimiter=",", fmt="%.9g")

            if self.state.save_png:
                fig_path = os.path.join(output_folder, f"{file_stub}.{ext}")
                self.save_map_figure(
                    arr,
                    map_name,
                    settings,
                    fig_path,
                    dpi=dpi,
                    mask_layers=mask_layers,
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

            file_stub = sanitize_filename(f"{numerator}_over_{denominator}")
            if prefix:
                file_stub = f"{prefix}_{file_stub}"

            if self.state.save_csv:
                csv_path = os.path.join(output_folder, f"{file_stub}.csv")
                np.savetxt(csv_path, ratio_arr, delimiter=",", fmt="%.9g")

            if self.state.save_png:
                fig_path = os.path.join(output_folder, f"{file_stub}.{ext}")
                self.save_map_figure(
                    ratio_arr,
                    ratio_name,
                    settings,
                    fig_path,
                    dpi=dpi,
                    mask_layers=mask_layers,
                )

        summary = self.build_summary_dict()
        base_name = prefix if prefix else "elementti_run"
        summary_path = os.path.join(output_folder, f"{base_name}_summary.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        methods_text = self.build_methods_text()
        methods_path = os.path.join(output_folder, "methods_Elementti.txt")

        with open(methods_path, "w", encoding="utf-8") as f:
            f.write(methods_text)

        return summary_path, methods_path


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to Elementti")
        self.setSubTitle(
            "A software for processing and visualization of SEM-EDS elemental maps, "
            "developed by the Laboratory of High Temperature Processes and Materials at "
            "Åbo Akademi University, Finland."
        )

        layout = QVBoxLayout()

        info = QLabel(
            "Citation:<br><br>"
            f"<b>{APP_CITATION_TEXT}</b><br><br>"
            "Click Next to begin."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)
        layout.addStretch()

        self.setLayout(layout)


class UploadPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 1: Upload your CSV files")
        self.setSubTitle("Choose one file or many files. When you are done, click Next.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- Click 'Choose CSV files'\n"
            "- Select one or many CSV files at once\n"
            "- You can click the button again if you want to add more files\n"
            "- Or click 'Try with sample data' to load three demo CSV files"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        self.choose_button = QPushButton("Choose CSV files")
        self.sample_button = QPushButton("Try with sample data")
        self.remove_button = QPushButton("Remove selected file")
        self.clear_button = QPushButton("Clear all")

        self.choose_button.clicked.connect(self.choose_files)
        self.sample_button.clicked.connect(self.load_sample_data)
        self.remove_button.clicked.connect(self.remove_selected_file)
        self.clear_button.clicked.connect(self.clear_all_files)

        row.addWidget(self.choose_button)
        row.addWidget(self.sample_button)
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

        self.setLayout(layout)

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
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_names:
            return

        added = 0
        for path in file_names:
            if path not in self.state.selected_file_paths:
                self.state.selected_file_paths.append(path)
                added += 1

        self.refresh_list()
        self.status_label.setText(
            f"{added} new file(s) added. Click Next when you are done."
            if added > 0 else "Those files were already added."
        )
        self.completeChanged.emit()

    def load_sample_data(self):
        try:
            if not self.state.sample_data_folder or not os.path.isdir(self.state.sample_data_folder):
                self.state.sample_data_folder = tempfile.mkdtemp(prefix="elementti_demo_")

            sample_paths = create_sample_data_files(self.state.sample_data_folder)

            added = 0
            for path in sample_paths:
                if path not in self.state.selected_file_paths:
                    self.state.selected_file_paths.append(path)
                    added += 1

            self.refresh_list()
            self.status_label.setText(
                f"{added} sample file(s) added. "
                f"The demo data include grayscale_demo.csv, element1_demo.csv, and element2_demo.csv "
                f"(384 rows × 512 columns)."
                if added > 0 else
                "Sample data were already added."
            )
            self.completeChanged.emit()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Could not create sample data",
                f"The app could not create the sample CSV files.\n\n{str(e)}"
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
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.state.selected_file_paths.clear()
            self.state.renamed_names_by_path.clear()
            self.state.file_processing_settings.clear()
            self.refresh_list()
            self.status_label.setText("All uploaded files were cleared.")
            self.completeChanged.emit()


class RenamePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 2: Rename your files")
        self.setSubTitle("Type the names you want the app to use later, then click Next.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- The left column shows the original CSV file name\n"
            "- Use the right column to enter the name that will be used in the app\n"
            "Examples: Cl, Fe, O, Cl500X, Fe100X, etc.\n"
            "Each name must be unique"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

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

        self.setLayout(layout)

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

        if len(names) != len(set(names)):
            QMessageBox.warning(self, "Duplicate names", "Each chosen name must be unique.")
            return False

        self.state.renamed_names_by_path = result
        self.status_label.setText("Renaming finished successfully.")
        return True


class OutputsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 3: Choose what you want to generate")
        self.setSubTitle("Choose which maps to plot and which ratios to create, then click Next.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- Tick the renamed files you want to plot as maps\n"
            "- Add any ratios you want, such as Cl / Fe or Ca / P\n"
            "- You can remove a ratio if you change your mind"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

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

        row = QHBoxLayout()
        self.remove_ratio_button = QPushButton("Remove selected ratio")
        self.remove_ratio_button.clicked.connect(self.remove_selected_ratio)
        row.addWidget(self.remove_ratio_button)
        row.addStretch()
        layout.addLayout(row)

        self.status_label = QLabel("Choose at least one map or one ratio.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        self.populate_maps_list()
        self.populate_ratio_combos()
        self.populate_ratio_list()
        self.update_status()

    def isComplete(self):
        return len(self.get_selected_maps()) > 0 or len(self.state.selected_ratios) > 0

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
        if ratio in self.state.selected_ratios:
            QMessageBox.warning(self, "Duplicate ratio", "That ratio has already been added.")
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

    def validatePage(self):
        self.state.selected_maps = self.get_selected_maps()
        if not self.state.selected_maps and not self.state.selected_ratios:
            QMessageBox.warning(self, "Nothing selected", "Please choose at least one map or one ratio.")
            return False
        return True

    def update_status(self):
        self.status_label.setText(
            f"Selected maps: {len(self.get_selected_maps())}    "
            f"Selected ratios: {len(self.state.selected_ratios)}"
        )


class ProcessingSettingsPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 4: Manual row/column removal and scale factor")
        self.setSubTitle("Write only the row and column numbers you want removed. Leave blank if you want to keep everything.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- All row and column numbers are 1-based\n"
            "- Write rows to remove like: 1,2,3,4,5,7\n"
            "- Write columns to remove like: 1,2,5\n"
            "- Leave the box blank if you do not want to remove anything\n"
            "- Then choose the scale factor to convert the imported values to the correct scale\n\n"
            "Typical values:\n"
            "- Greyscale image: scale factor = 1.0\n"
            "- Elemental map stored as AT% x 100: scale factor = 0.01"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            [
                "Renamed file",
                "Rows to remove",
                "Columns to remove",
                "Scale factor",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.status_label = QLabel("Set the rows, columns, and scale factor for each file, then click Next.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

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
                    manual_rows="",
                    manual_columns="",
                    scale_factor=self.default_scale_for_name(name),
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
            rows_edit.setPlaceholderText("Example: 1,2,3,4,5,7")
            self.table.setCellWidget(row, 1, rows_edit)

            cols_edit = QLineEdit(settings.manual_columns)
            cols_edit.setPlaceholderText("Example: 1,2,5")
            self.table.setCellWidget(row, 2, cols_edit)

            scale_edit = QLineEdit(settings.scale_factor)
            self.table.setCellWidget(row, 3, scale_edit)

    def validatePage(self):
        result: Dict[str, FileReadSettings] = {}

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            path = name_item.data(Qt.UserRole)
            renamed = name_item.text()

            manual_rows = self.table.cellWidget(row, 1).text().strip()
            manual_columns = self.table.cellWidget(row, 2).text().strip()
            scale_text = self.table.cellWidget(row, 3).text().strip()

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
                float(scale_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid scale factor", f"Scale factor for '{renamed}' must be a number.")
                return False

            result[path] = FileReadSettings(
                manual_rows=manual_rows,
                manual_columns=manual_columns,
                scale_factor=scale_text,
            )

        self.state.file_processing_settings = result
        return True


class MaskingIntroPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 5: Masking and noise removal")
        self.setSubTitle("Decide whether you want to apply masking and noise-removal rules before visualization and export.")

        layout = QVBoxLayout()

        info = QLabel(
            "Why this may be necessary:\n"
            "- Raw SEM-EDS data may contain background regions, pores, cracks, shadowed areas,\n"
            "  detector-related artifacts, or pixels with very weak signal intensity\n"
            "- Such pixels may reduce map readability\n"
            "- They may also create unstable or misleading ratio values, especially when denominator values are very small\n\n"
            "Available methods in the next page:\n"
            "1. Grayscale masking, using one or more grayscale rules with separate colors\n"
            "2. Low-value protection, by replacing very small values with a chosen minimum\n"
            "3. Conditional replacement, for rules such as: if Fe > 90, then set Cl = 0\n\n"
            "Do you want to apply masking and noise removal?"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Apply masking and noise removal:"))
        self.enable_combo = QComboBox()
        self.enable_combo.addItems(["No", "Yes"])
        row.addWidget(self.enable_combo)
        row.addStretch()
        layout.addLayout(row)

        self.setLayout(layout)

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
        self.setSubTitle("Choose one, two, or all three methods, then click Next.")

        layout = QVBoxLayout()

        info = QLabel(
            "- All thresholds, limits, and replacement values are interpreted AFTER scaling\n"
            "- You may use one method or multiple methods\n"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(QLabel("Method 1: Grayscale masking"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Use this method:"))
        self.grayscale_enable_combo = QComboBox()
        self.grayscale_enable_combo.addItems(["No", "Yes"])
        self.grayscale_enable_combo.currentTextChanged.connect(self.update_enabled_state)
        row.addWidget(self.grayscale_enable_combo)
        row.addStretch()
        layout.addLayout(row)

        grayscale_info = QLabel(
            "You may add multiple grayscale masking rules, for example:\n"
            "- mask values below 20 with black, mask values above 240 with white, or mask values outside 20 and 240 with gray\n"
        )
        grayscale_info.setWordWrap(True)
        layout.addWidget(grayscale_info)

        self.grayscale_table = QTableWidget(0, 6)
        self.grayscale_table.setHorizontalHeaderLabels(
            [
                "Use rule",
                "Source file",
                "Condition",
                "Value A",
                "Value B",
                "Mask color",
            ]
        )
        self.grayscale_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.grayscale_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
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

        conditional_info = QLabel(
            "Define rules such as:\n"
            "- If Fe > 90, then set Cl = 0\n"
            "- If Grey < 20, then set O = 0\n"
            "- If Ca is between 10 and 25, then set P = 5"
        )
        conditional_info.setWordWrap(True)
        layout.addWidget(conditional_info)

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

        self.setLayout(layout)

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
        rules = self.state.masking_and_noise_settings.grayscale_rules

        for rule in rules:
            self.add_grayscale_rule_row(rule, names)

    def add_grayscale_rule_row(self, rule: Optional[GrayscaleMaskRule] = None, names: Optional[List[str]] = None):
        if names is None:
            names = self.state.get_ordered_names()

        if rule is None:
            source_default = names[0] if names else ""
            rule = GrayscaleMaskRule(
                enabled=True,
                source_name=source_default,
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

        condition_combo = QComboBox()
        condition_combo.addItems(["below", "above", "between", "outside"])
        condition_combo.setCurrentText(rule.condition or "below")
        condition_combo.currentTextChanged.connect(lambda _text, r=row: self.update_grayscale_row_state(r))
        self.grayscale_table.setCellWidget(row, 2, condition_combo)

        value_a_edit = QLineEdit(rule.value_a)
        value_a_edit.setPlaceholderText("Example: 20")
        self.grayscale_table.setCellWidget(row, 3, value_a_edit)

        value_b_edit = QLineEdit(rule.value_b)
        value_b_edit.setPlaceholderText("Only for between/outside")
        self.grayscale_table.setCellWidget(row, 4, value_b_edit)

        color_edit = QLineEdit(rule.color)
        color_edit.setPlaceholderText("Example: black or #000000")
        self.grayscale_table.setCellWidget(row, 5, color_edit)

        self.update_grayscale_row_state(row)

    def read_grayscale_table_to_settings(self):
        rules = []
        for row in range(self.grayscale_table.rowCount()):
            use_combo = self.grayscale_table.cellWidget(row, 0)
            source_combo = self.grayscale_table.cellWidget(row, 1)
            condition_combo = self.grayscale_table.cellWidget(row, 2)
            value_a_edit = self.grayscale_table.cellWidget(row, 3)
            value_b_edit = self.grayscale_table.cellWidget(row, 4)
            color_edit = self.grayscale_table.cellWidget(row, 5)

            rules.append(
                GrayscaleMaskRule(
                    enabled=(use_combo.currentText() == "Yes"),
                    source_name=source_combo.currentText().strip(),
                    condition=condition_combo.currentText().strip(),
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
        condition_combo = self.grayscale_table.cellWidget(row, 2)
        value_a_edit = self.grayscale_table.cellWidget(row, 3)
        value_b_edit = self.grayscale_table.cellWidget(row, 4)
        color_edit = self.grayscale_table.cellWidget(row, 5)

        rule_enabled = method_enabled and use_combo.currentText() == "Yes"
        needs_two_values = condition_combo.currentText() in ("between", "outside")

        use_combo.setEnabled(method_enabled)
        source_combo.setEnabled(rule_enabled)
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
                condition_operator=">",
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
        operator_combo.addItems([">", ">=", "<", "<=", "between"])
        operator_combo.setCurrentText(rule.condition_operator or ">")
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
                    condition_operator=operator_combo.currentText().strip(),
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
                condition_operator=">",
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

            for idx, rule in enumerate(active_grayscale_rules, start=1):
                if rule.source_name.strip() == "":
                    QMessageBox.warning(self, "Missing grayscale source", f"Grayscale rule {idx}: please choose the source file.")
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

                if rule.condition in ("below", "above"):
                    if rule.value_a.strip() == "":
                        QMessageBox.warning(self, "Missing threshold", f"Grayscale rule {idx}: please enter Value A.")
                        return False
                    try:
                        float(rule.value_a)
                    except ValueError:
                        QMessageBox.warning(self, "Invalid threshold", f"Grayscale rule {idx}: Value A must be a number.")
                        return False

                elif rule.condition in ("between", "outside"):
                    if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                        QMessageBox.warning(
                            self,
                            "Missing range values",
                            f"Grayscale rule {idx}: '{rule.condition}' requires both Value A and Value B."
                        )
                        return False
                    try:
                        float(rule.value_a)
                        float(rule.value_b)
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            "Invalid range values",
                            f"Grayscale rule {idx}: Value A and Value B must be numbers."
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
                    floor_value = float(floor_text)
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
                    float(rule.replacement_value)
                except ValueError:
                    QMessageBox.warning(self, "Invalid replacement value", f"Rule {idx}: replacement value must be a number.")
                    return False

                if rule.condition_operator == "between":
                    if rule.value_a.strip() == "" or rule.value_b.strip() == "":
                        QMessageBox.warning(self, "Missing condition values", f"Rule {idx}: 'between' requires both Value A and Value B.")
                        return False
                    try:
                        float(rule.value_a)
                        float(rule.value_b)
                    except ValueError:
                        QMessageBox.warning(self, "Invalid condition values", f"Rule {idx}: Value A and Value B must be numbers.")
                        return False
                else:
                    if rule.value_a.strip() == "":
                        QMessageBox.warning(self, "Missing condition value", f"Rule {idx}: please enter Value A.")
                        return False
                    try:
                        float(rule.value_a)
                    except ValueError:
                        QMessageBox.warning(self, "Invalid condition value", f"Rule {idx}: Value A must be a number.")
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
        self.setSubTitle("The current min and max below are based on the processed values the app will use.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- Each row is one output you selected in Step 3\n"
            "- Choose either Continuous or Manual bins\n"
            "- Choose the guide bar side: left or right\n"
            "- Write the guide bar label text\n"
            "- 'Processed min' and 'Processed max' are calculated after scaling and after any active masking/noise-removal rules\n"
            "- In Continuous mode, you may choose a standard colormap or a single-color gradient such as green, red, or blue\n"
            "- Example: choosing green means low values appear dark green and high values appear bright green\n"
            "- If you choose Manual bins, write the boundary numbers separated by commas\n"
            "- Example boundaries: 0, 0.5, 1, 1.5, 2\n"
            "- These boundaries create these bins: [0,0.5], [0.5,1], [1,1.5], [1.5,2]\n"
            "- For Manual bins, write one color per bin separated by commas\n"
            "- You can use names or hex colors\n\n"
            "More color examples:\n"
            "white, black, gray, lightgray, silver, red, darkred, firebrick, orange, gold,\n"
            "yellow, khaki, green, lime, forestgreen, olive, cyan, turquoise, teal,\n"
            "blue, navy, royalblue, deepskyblue, purple, violet, magenta, pink,\n"
            "brown, maroon\n\n"
            "Hex examples:\n"
            "#FFFFFF, #000000, #FF0000, #00FF00, #0000FF, #FFA500"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

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
        layout.addWidget(self.table)

        self.status_label = QLabel("Choose the display settings you want, then click Next.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 7: Choose display settings")
        else:
            self.setTitle("Step 6: Choose display settings")

        try:
            self.populate_table()
            self.status_label.setText("Processed min/max were calculated successfully.")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Could not calculate processed min/max",
                f"The app could not read one of the files with the current settings.\n\n{str(e)}"
            )
            self.status_label.setText("There was a problem reading one of the files. Please go back and check earlier steps.")

    def get_output_names(self) -> List[str]:
        outputs = list(self.state.selected_maps)
        outputs.extend(f"{num} / {den}" for num, den in self.state.selected_ratios)
        return outputs

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
        stats = self.state.processed_min_max_by_name
        for output_name in self.get_output_names():
            if output_name in self.state.display_settings:
                continue

            min_text = ""
            max_text = ""
            if output_name in stats:
                min_val, max_val = stats[output_name]
                if min_val is not None and max_val is not None:
                    min_text = f"{min_val:g}"
                    max_text = f"{max_val:g}"

            self.state.display_settings[output_name] = OutputDisplaySettings(
                mode="Continuous",
                colormap=self.default_colormap_for_output(output_name),
                colorbar_side="right",
                colorbar_label=default_colorbar_label(output_name),
                display_min=min_text,
                display_max=max_text,
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
            self.table.setCellWidget(row, 2, cmap_combo)

            side_combo = QComboBox()
            side_combo.addItems(["right", "left"])
            side_combo.setCurrentText(settings.colorbar_side)
            self.table.setCellWidget(row, 3, side_combo)

            self.table.setCellWidget(row, 4, QLineEdit(settings.colorbar_label))

            min_item = QTableWidgetItem("" if min_val is None else f"{min_val:g}")
            min_item.setFlags(min_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, min_item)

            max_item = QTableWidgetItem("" if max_val is None else f"{max_val:g}")
            max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, max_item)

            display_min_edit = QLineEdit(settings.display_min)
            display_min_edit.setPlaceholderText("use processed min")
            self.table.setCellWidget(row, 7, display_min_edit)

            display_max_edit = QLineEdit(settings.display_max)
            display_max_edit.setPlaceholderText("use processed max")
            self.table.setCellWidget(row, 8, display_max_edit)

            bins_edit = QLineEdit(settings.bins)
            bins_edit.setPlaceholderText("Example: 0, 0.5, 1, 1.5, 2")
            self.table.setCellWidget(row, 9, bins_edit)

            bin_colors_edit = QLineEdit(settings.bin_colors)
            bin_colors_edit.setPlaceholderText("Example: blue, green, yellow, red or #FF0000")
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

    def validate_number_or_blank(self, value_text: str, field_name: str, output_name: str) -> bool:
        if value_text.strip() == "":
            return True
        try:
            float(value_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid number", f"{field_name} for '{output_name}' must be a number or left blank.")
            return False
        return True

    def validate_bins(self, bins_text: str, output_name: str) -> bool:
        if bins_text.strip() == "":
            QMessageBox.warning(self, "Missing bins", f"Please enter manual bins for '{output_name}', or change its mode.")
            return False

        parts = [p.strip() for p in bins_text.split(",")]
        if len(parts) < 2:
            QMessageBox.warning(self, "Not enough bins", f"Manual bins for '{output_name}' must contain at least two numbers.")
            return False

        try:
            values = [float(part) for part in parts]
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

            if display_min_text and display_max_text and float(display_min_text) >= float(display_max_text):
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

        self.state.display_settings = result
        return True


class GeneratePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 8: Choose output folder and generate")
        self.setSubTitle("Choose where the app should save the results, then click Generate.")

        layout = QVBoxLayout()

        info = QLabel(
            "What to do here:\n"
            "- Choose the output folder where the results will be saved\n"
            "- You can give this run a project name if you want\n"
            "- Choose whether to save figures and/or processed CSV files\n"
            "- Choose image format and DPI\n"
            "- When you click Generate, the app will save the outputs and stay open so you can go back and revise settings"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

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

        self.citation_title_label = QLabel("Citation:")
        layout.addWidget(self.citation_title_label)

        self.citation_label = QLabel(APP_CITATION_TEXT)
        self.citation_label.setWordWrap(True)
        self.citation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.citation_label)

        self.setLayout(layout)

    @property
    def state(self) -> AppState:
        return self.wizard().state

    def initializePage(self):
        if self.state.masking_and_noise_settings.enabled:
            self.setTitle("Step 8: Choose output folder and generate")
        else:
            self.setTitle("Step 7: Choose output folder and generate")

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
            self.output_folder_edit.text().strip() or ""
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
            QMessageBox.warning(self, "Cannot create folder", f"The app could not create or access the output folder.\n\n{str(e)}")
            return False

        return True

    def update_summary(self):
        folder = self.output_folder_edit.text().strip()
        fig_text = "Yes" if self.save_png_checkbox.isChecked() else "No"
        csv_text = "Yes" if self.save_csv_checkbox.isChecked() else "No"
        fmt = self.figure_format_combo.currentText().strip()
        dpi = self.dpi_edit.text().strip() or "(not set yet)"
        masking_text = "Yes" if self.state.masking_and_noise_settings.enabled else "No"

        self.summary_label.setText(
            f"Summary:\n"
            f"- Selected maps: {len(self.state.selected_maps)}\n"
            f"- Selected ratios: {len(self.state.selected_ratios)}\n"
            f"- Masking and noise removal enabled: {masking_text}\n"
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
    PAGE_GENERATE = 8

    def __init__(self):
        super().__init__()

        self.state = AppState()
        self.engine = ProcessingEngine(self.state)

        self.setWindowTitle("Elementti")
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
        self.setPage(self.PAGE_GENERATE, GeneratePage())

        self.setStartId(self.PAGE_WELCOME)

        if self.layout() is not None:
            self.layout().setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        self.setMinimumSize(950, 650)
        self.setMaximumSize(16777215, 16777215)
        self.resize(1200, 780)

        self.setButtonText(QWizard.FinishButton, "Generate")
        self.setButtonText(QWizard.CancelButton, "Close")

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


app = QApplication(sys.argv)
wizard = ElementtiWizard()
wizard.show()
sys.exit(app.exec())

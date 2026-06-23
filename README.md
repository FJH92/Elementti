# Elementti

**Current version:** v1.0.9
**SoftwareX published version:** v1.0.0
**Latest version file:** `Elementti_1_0_9.py`

Elementti is a Python-based desktop graphical user interface (GUI) for processing and visualization of SEM-EDS elemental maps exported in CSV format.

The version associated with the published SoftwareX article is **Elementti v1.0.0**. That release is preserved for reproducibility. **Elementti v1.0.9** is the current updated post-publication version.

## Citation

If you use Elementti, please cite:

Jafarihonar, F., & Vainio, E. (2026). Elementti: A Python-based desktop GUI application for processing and visualization of SEM-EDS elemental maps. SoftwareX, 35, 102826. https://doi.org/10.1016/j.softx.2026.102826

## Overview

Elementti was developed to support more consistent and reproducible post-processing of SEM-EDS elemental maps. The software combines file import, preprocessing, ratio-map generation, formula-map generation, masking, visualization, line-profile analysis, and export within a single guided workflow. It is intended for users who work with SEM-EDS map data in CSV format and want an accessible, vendor-independent way to prepare processed data and figure-ready outputs.

## Main features

* import one or multiple CSV map files
* assign user-defined names to imported files
* remove unwanted header rows and columns
* apply file-specific scaling factors
* detect or enter pixel size using common units such as nm, µm/um, mm, cm, or m
* generate direct elemental maps and derived ratio maps
* generate custom formula maps using safe expression parsing
* apply grayscale masking
* apply low-value protection
* apply conditional replacement rules
* visualize outputs using continuous colormaps or manually defined bins
* adjust display minimum and maximum values for color-scale control
* round displayed data minimum and maximum values for readability
* control colorbar tick-number size and colorbar label-text size per output
* add scale bars when pixel-size information is available
* create line profiles with optional physical-distance scaling
* edit or clear the line-profile x-axis label
* use dual y-axes for mixed element and ratio line profiles
* set optional manual y-axis limits for line profiles
* control line-profile tick size and text size for axis numbers, labels, title, legend, and map-panel labels
* choose line-profile legend placement inside the plot or outside right
* export processed CSV files and figure files
* automatically save selected processing and visualization settings in a JSON summary file
* generate a plain-text methods file for record keeping and reproducibility
* load built-in fixed demo data for testing and demonstration

## Repository contents

* `Elementti_1_0_9.py` – latest source file
* `sample_data/` – example CSV files for demonstration
* `requirements.txt` – required Python packages
* `LICENSE` – software license
* `README.md` – project description and usage instructions

## Requirements

Elementti is implemented in Python and uses the following main packages:

* PySide6
* NumPy
* Matplotlib

## Installation

1. Install Python 3.
2. Clone or download this repository.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, on Windows:

```bash
py -m pip install numpy matplotlib PySide6
```

## How to run

Run the program with:

```bash
python Elementti_1_0_9.py
```

On Windows, this may also be run with:

```bash
py Elementti_1_0_9.py
```

## Windows executable

For users who do not want to run the Python source code directly, a Windows executable may be provided as a GitHub release asset.

Use the GitHub Releases page to download the executable for the latest version when available.

## Demo data

Elementti includes built-in demo data for testing and demonstration.

Since v1.0.8, the demo data are fixed and repeatable by default. The generated demo CSV files include metadata rows and detectable pixel-size information.

Always check the detected pixel size before final export.

## Typical workflow

A typical workflow in Elementti consists of the following steps:

1. import CSV files
2. assign names to the imported files
3. select direct outputs, ratio maps, or formula maps
4. define preprocessing settings
5. optionally apply masking and rule-based processing
6. choose visualization settings
7. optionally define scale bars or line profiles
8. adjust colorbar, tick-size, text-size, and line-profile labeling settings as needed
9. export processed data and figure files

## Output files

Depending on the selected options, Elementti can export:

* processed CSV files
* figure files
* a JSON summary file containing selected processing, visualization, size, label, and line-profile settings
* a plain-text methods file summarizing the workflow

## Notes on visualization

Display minimum and maximum values change the color scale; they do not remove data.

Values above the display maximum use the top color, and values below the display minimum use the bottom color.

Data min and Data max are displayed rounded to at most four decimals. Disp min and Disp max accept values with at most four decimals.

## Main changes in v1.0.9

Elementti v1.0.9 is a post-publication update. Main updates since the original release include:

* unit-aware pixel-size handling
* custom formula maps
* improved formula-map interface and instructions
* improved line-profile tools
* optional physical-distance scaling for line-profile x-axes
* dual y-axes for mixed element and ratio profiles
* optional manual y-axis limits for line profiles
* clearer colorbar and guide-bar endpoint labels
* improved line-profile legend placement
* fixed and repeatable demo data
* safer export filenames to prevent accidental overwriting
* improved Windows executable icon handling
* updated in-app citation to the published SoftwareX reference
* clickable DOI shown on its own line in the in-app citation
* per-output Tick size and Text size controls for colorbar numbers and labels
* matching Tick size and Text size controls for line-profile plots and map-panel labels
* editable line-profile x-axis label, with clearing supported
* shorter display-settings column headers for readability
* rounded Data min and Data max display values
* four-decimal input limit for Disp min and Disp max
* more compact display-settings columns to preserve width for Bins and Bin colors
* recording of new size, label, and x-axis-label settings in the JSON summary and methods text
* updated icon and welcome-image resource names to `icoElementti_1_0_9.ico` and `pngElementti_1_0_9.png`, with older resource names still supported as fallbacks

## Intended use

Elementti is intended for post-export processing and visualization of SEM-EDS elemental maps stored in CSV format. The software is designed to support reproducible and consistent preparation of elemental maps, ratio maps, formula maps, visualizations, and line profiles for interpretation, comparison, and figure generation.

## Development

Elementti was developed in the High-Temperature Processes and Materials Group at Åbo Akademi University, Turku, Finland.

## Status

This repository contains the Elementti software associated with the SoftwareX article.

Version **v1.0.0** corresponds to the published SoftwareX version and is preserved for reproducibility.

Version **v1.0.9** is the current updated post-publication version and includes corrections, usability improvements, additional functionality, updated citation handling, expanded size and label controls, and improved export traceability.

## Author

Farzad Jafarihonar
Åbo Akademi University
Turku, Finland

## License

Elementti is distributed under the GPL-3.0 license. See `LICENSE` for details.

# Elementti

**Current version:** v1.1.1
**SoftwareX published version:** v1.0.0
**Latest version file:** `src/elementti.py`

Elementti is a Python-based desktop graphical user interface (GUI) for processing and visualization of SEM-EDS elemental maps exported in CSV format.

The version associated with the published SoftwareX article is **Elementti v1.0.0**. That release is preserved for reproducibility. **Elementti v1.1.1** is the current updated post-publication version.

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
* draw regions on a map (rectangle, square, ellipse, polygon, or freehand) and compare the mean composition inside each region, with an optional zoomed detail area
* save area-analysis results as CSV, figures, and an Excel report
* save the current map, chart, or CSV directly from the Display, Line profile, and Area analysis pages
* copy display settings from one output to others with group formatting
* review suggested acquisition groups and import settings before applying them
* verify input files with SHA-256 fingerprints and review changed, missing, or unverified source files
* export processed CSV files and figure files
* save every setting in a project file (`Project.elementti.json`) and open it again later, with the same map files or with new ones
* write `Detailed explanations.docx`, a Word document that explains every step of the run with formulas and tables and ends with a methods-section description
* read **How it works** explanations of each method, with formulas and worked examples
* see all nine steps in a step list and click back to any finished step
* build a stand-alone `Elementti.exe` that runs without Python
* load built-in fixed demo data for testing and demonstration

## Repository contents

* `src/elementti.py` – latest source file (one file; the logo and icon are built in)
* `sample_data/` – example CSV files for demonstration
* `requirements.txt` – required Python packages
* `LICENSE` – software license
* `README.md` – project description and usage instructions
* `CHANGELOG` – changes in each version
* `CITATION.md` – how to cite Elementti

## Requirements

Elementti is implemented in Python and uses the following main packages:

* PySide6
* NumPy
* Matplotlib
* openpyxl (needed only for Excel reports)

## Installation

1. Install Python 3.
2. Clone or download this repository.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, on Windows:

```bash
py -m pip install numpy matplotlib PySide6 openpyxl
```

## How to run

Run the program with:

```bash
python src/elementti.py
```

On Windows, this may also be run with:

```bash
py src/elementti.py
```

To check the installation without opening a window, run the built-in demo check. It makes the demo outputs in a temporary folder and lists them:

```bash
python src/elementti.py --check-demo
```

## Windows executable

For users who do not want to run the Python source code directly, a Windows executable may be provided as a GitHub release asset.

Use the GitHub Releases page to download the executable for the latest version when available.

You can also build the executable yourself. Start Elementti from the `.py` file and click **Create Elementti.exe** on the welcome page, or run:

```bash
python src/elementti.py --make-exe
```

The program file is saved next to the source file, or in a folder given after `--make-exe`. PyInstaller is installed automatically the first time, which needs an internet connection once. On Windows, the program is signed automatically if a code-signing certificate is installed; otherwise it is left unsigned.

## Demo data

Elementti includes built-in demo data for testing and demonstration.

Since v1.0.8, the demo data are fixed and repeatable by default. The generated demo CSV files include metadata rows and detectable pixel-size information.

Always check the detected pixel size before final export.

## Projects

Every time you click **Generate**, Elementti saves a project file, `Project.elementti.json`, together with the outputs. It records every setting of every step and a fingerprint of every input file, so Elementti can tell if a file has changed since the project was saved.

To use it again, click **Open project** in step 1 and choose:

* **Its own map files** to make the same outputs again. If the map files were moved, Elementti asks for the folder that contains them.
* **Other map files** to use the same settings with new maps, for example another sample measured the same way. Files with identical names are matched automatically; every other map is chosen in a matching dialog.

Then click **Next** through the steps. Every setting is already filled in, and you can check or change it.

## Typical workflow

A typical workflow in Elementti consists of the following steps:

1. import CSV files
2. assign names to the imported files
3. select direct outputs, ratio maps, or formula maps
4. define preprocessing settings
5. optionally apply masking and rule-based processing
6. choose visualization settings
7. optionally define scale bars or line profiles
8. optionally draw regions for area analysis
9. adjust colorbar, tick-size, text-size, and line-profile labeling settings as needed
10. export processed data and figure files

## Output files

Depending on the selected options, Elementti can export:

* processed CSV files
* figure files
* area-analysis tables, region outlines, figures, and an Excel report, when area analysis is used
* `Project.elementti.json`, a project file with every setting, which can be opened again with **Open project**
* `Detailed explanations.docx` (optional), a Word document that explains every step of the run in words, formulas, tables, maps, and charts, including what each rule changed and which source files were used, and ends with a description of the run that can be adapted for the methods section of a paper

## Notes on visualization

Display minimum and maximum values change the color scale; they do not remove data.

Values above the display maximum use the top color, and values below the display minimum use the bottom color.

Data min and Data max are displayed rounded to at most four decimals. Disp min and Disp max accept values with at most four decimals.

## Main changes in v1.1.1

Elementti v1.1.1 is a post-publication update. Main updates since v1.1.0:

* a new optional **Area analysis** step for comparing the composition of regions drawn on a map, with zoom and pan, a detail area, CSV and figure outputs, and an Excel report
* quick export buttons for saving the current map, chart, or CSV from the Display, Line profile, and Area analysis pages
* group formatting for copying display settings between outputs
* review dialogs for suggested acquisition groups and import settings
* source-file verification with SHA-256 fingerprints, recorded in the project file and the Word report
* an option to skip the Word report
* project file format version 4; projects saved with v1.1.0 still open

## Main changes in v1.1.0

Main updates in v1.1.0 compared with v1.0.9:

* project files (`Project.elementti.json`) that save every setting, and an **Open project** button to reuse them with the same or new map files
* `Detailed explanations.docx`, which replaces the JSON summary and plain-text methods file
* **How it works** windows with formulas and worked examples for each method
* a step list for moving between the eight steps
* a progress window while outputs are generated
* a **Create Elementti.exe** button and a `--make-exe` command-line option for building a stand-alone program
* a `--check-demo` command-line option for a quick check without a window
* the logo and icon built into the source file, so one `.py` file is all that is needed

See `CHANGELOG` for the full list of changes in every version.

## Intended use

Elementti is intended for post-export processing and visualization of SEM-EDS elemental maps stored in CSV format. The software is designed to support reproducible and consistent preparation of elemental maps, ratio maps, formula maps, visualizations, and line profiles for interpretation, comparison, and figure generation.

## Development

Elementti was developed in the High-Temperature Processes and Materials Group at Åbo Akademi University, Turku, Finland.

## Status

This repository contains the Elementti software associated with the SoftwareX article.

Version **v1.0.0** corresponds to the published SoftwareX version and is preserved for reproducibility.

Version **v1.1.1** is the current updated post-publication version. It includes corrections, usability improvements, and additional functionality, including area analysis, project files with source-file verification, detailed explanations of each run, and a built-in way to create a stand-alone executable.

## Author

Farzad Jafarihonar
Åbo Akademi University
Turku, Finland

## License

Elementti is distributed under the GPL-3.0 license. See `LICENSE` for details.

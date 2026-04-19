# Elementti

**Current version:** v1.0.0

Elementti is a Python-based desktop graphical user interface (GUI) for processing and visualization of SEM-EDS elemental maps exported in CSV format.

## Overview

Elementti was developed to support more consistent and reproducible post-processing of SEM-EDS elemental maps. The software combines file import, preprocessing, ratio-map generation, masking, visualization, and export within a single guided workflow. It is intended for users who work with SEM-EDS map data in CSV format and want an accessible, vendor-independent way to prepare processed data and figure-ready outputs.

## Main features

* import one or multiple CSV map files
* assign user-defined names to imported files
* remove unwanted header rows and columns
* apply file-specific scaling factors
* generate direct elemental maps and derived ratio maps
* apply grayscale masking
* apply low-value protection
* apply conditional replacement rules
* visualize outputs using continuous colormaps or manually defined bins
* export processed CSV files and figure files
* automatically save selected processing and visualization settings in a JSON summary file
* generate a plain-text methods file for record keeping and reproducibility
* load built-in reproducible or random sample data for testing and demonstration

## Repository contents

* `src/elementti.py` – main source code
* `sample_data/` – example CSV files for demonstration
* `requirements.txt` – required Python packages
* `LICENSE.txt` – software license
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

## How to run

Run the program with:

```bash
python src/elementti.py
```

## Sample data

Elementti includes built-in sample data for testing and demonstration.

The software provides two sample-data options:

* **reproducible sample data** – generated using a fixed random seed
* **random sample data** – generated using a new random seed

When sample data are used, the selected mode and seed are recorded in the exported summary information.

## Typical workflow

A typical workflow in Elementti consists of the following steps:

1. import CSV files
2. assign names to the imported files
3. select direct outputs and ratio maps
4. define preprocessing settings
5. optionally apply masking and rule-based processing
6. choose visualization settings
7. export processed data and figure files

## Output files

Depending on the selected options, Elementti can export:

* processed CSV files
* figure files
* a JSON summary file containing selected processing and visualization settings
* a plain-text methods file summarizing the workflow

## Intended use

Elementti is intended for post-export processing and visualization of SEM-EDS elemental maps stored in CSV format. The software is designed to support reproducible and consistent preparation of elemental maps and ratio maps for interpretation, comparison, and figure generation.

## Development

Elementti was developed in the High-Temperature Processes and Materials Group at Åbo Akademi University, Turku, Finland.

## Citation

If you use Elementti in your work, please cite the associated SoftwareX article and/or the Elementti software release.

Full bibliographic details will be added after publication.

## Status

This repository contains the Elementti software associated with the SoftwareX manuscript.
The current release corresponding to the submitted manuscript is **v1.0.0**.

The repository may continue to be updated during review and revision, while versioned releases preserve specific software snapshots.

## Author

Farzad Jafarihonar
Åbo Akademi University
Turku, Finland

## License

Elementti is distributed under the GPL-3.0 license. See `LICENSE.txt` for details.

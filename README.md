# Elementti

Elementti is a Python-based graphical user interface for processing and visualization of SEM-EDS elemental maps exported in CSV format.

## Overview

Elementti was developed to simplify routine post-processing of SEM-EDS elemental maps by combining file import, preprocessing, ratio-map generation, masking, visualization, and export within a single workflow. The software is intended for users who work with SEM-EDS map data in CSV format and want an accessible and reproducible way to prepare processed data and figures.

## Main features

- import one or multiple CSV map files
- assign user-defined names to imported files
- remove unwanted header rows and columns
- apply file-specific scaling factors
- generate elemental ratio maps
- apply optional masking rules
- apply low-value protection
- apply conditional replacement rules
- visualize maps using continuous colormaps or manual bins
- export processed arrays and figure files
- automatically save processing and visualization settings in a JSON summary file
- generate a plain-text methods file for record keeping

## Repository contents

- `src/elementti.py` – main source code
- `sample_data/` – sample CSV files for demonstration
- `requirements.txt` – required Python packages
- `LICENSE.txt` – software license
- `README.md` – repository description and usage instructions

## Requirements

Elementti is implemented in Python and uses the following main packages:

- PySide6
- NumPy
- Matplotlib

## Installation

1. Install Python 3 on your computer.
2. Download or clone this repository.
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

Sample CSV files for testing and demonstration are included in the `sample_data` folder.

## Typical workflow

Elementti is designed as a guided workflow. A typical use case includes:

1. import CSV files
2. assign names to the imported maps
3. select direct outputs and ratio maps
4. define preprocessing settings
5. optionally apply masking or other rule-based processing
6. choose visualization settings
7. export processed data and figure files

## Output files

Depending on the selected options, Elementti can export:

- processed CSV files
- figure files
- a JSON summary file containing selected processing and visualization settings
- a plain-text methods file summarizing the workflow

## Intended use

Elementti is intended for processing and visualization of SEM-EDS elemental maps exported in CSV format. The software is designed to support reproducible and consistent post-processing of elemental mapping data in microscopy-related applications.

## Development

Elementti was developed in the High-Temperature Processes and Materials Group at Åbo Akademi University, Turku, Finland.

## Citation



Full bibliographic details will be added after publication.

## Status

This repository contains the Elementti software described in the manuscript and may continue to be updated during manuscript preparation and revision.

## Author

Farzad Jafarihonar  
Åbo Akademi University  
Turku, Finland

## License

This software is distributed under the license provided in `LICENSE.txt`.

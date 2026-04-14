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

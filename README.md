# Computer Vision Project

This project implements various computer vision algorithms focusing on connected components analysis and Hough circle transformation. It provides tools for image processing and feature detection using Python and OpenCV.

## Features

- **Connected Components Analysis**: Implementation of connected components labeling algorithm for image segmentation and object detection
- **Hough Circle Transformation**: Detection of circular patterns in images using the Hough transform algorithm

## Project Structure

```
.
├── datasets/               # Input images for processing
│   ├── connected_comps.png
│   ├── image_hough.png
│   └── image_hough_small.png
├── docs/                  # Documentation files
├── functions/             # Core implementation of algorithms
│   ├── connected_components_func.py
│   └── hough_transformation_func.py
├── maincodes/            # Main execution scripts
│   ├── connected_comps.py
│   └── hough_circle.py
├── models/               # Model files (if any)
├── playground/           # Jupyter notebooks for experimentation
│   ├── connected_comps.ipynb
│   └── hough_circle.ipynb
├── results/              # Output directory for processed images
└── requirement.txt       # Project dependencies
```

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd cv-project
```

2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

Required dependencies:
- opencv-python
- matplotlib
- numpy

## Usage

### Connected Components Analysis

Run the connected components analysis on images:

```bash
python maincodes/connected_comps.py
```

This script processes images using connected components labeling to identify and label connected regions in binary images.

### Hough Circle Detection

Execute the Hough circle detection:

```bash
python maincodes/hough_circle.py
```

This script implements the Hough transform algorithm to detect circular patterns in images.

## Development

The project includes Jupyter notebooks in the `playground/` directory for experimentation and development:

- `connected_comps.ipynb`: Interactive notebook for connected components analysis
- `hough_circle.ipynb`: Interactive notebook for Hough circle transformation

## Results

Processed images and analysis results are saved in the `results/` directory.

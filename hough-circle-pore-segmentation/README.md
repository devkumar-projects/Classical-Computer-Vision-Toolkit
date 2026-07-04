# Pore Segmentation in Metal Foam via Hough Transform

Detecting and quantifying pores in metal-foam CT-scan slices using a **constrained circular Hough transform**, with custom duplicate-removal and porosity-estimation steps built on top of the base `scikit-image` pipeline.

## Problem

Given CT-scan volumes (`.tif`) of a porous metal foam sample, the goal is to:
1. Tune Hough-transform parameters to get a clean, non-redundant circle detection;
2. Report the number of detected pores per slice and their radius/diameter statistics;
3. Reduce over-segmentation (multiple overlapping circles on the same cavity);
4. Estimate the surface porosity of each slice from the retained circles.

## Approach

### 1. Constrained circular Hough transform
Rather than using an unconstrained Hough transform (which produces many redundant circles per cavity), detection is tuned with:
- Canny edge detection with `sigma=2.5` (stronger smoothing than default, to reduce noise-driven false edges)
- a restricted radius search range (8–70 px, step 2)
- a minimum distance between accepted peaks (12 px in both x and y)
- a relative threshold on the Hough accumulator (0.35 × max response) to drop weak candidates
- an extra, stricter confidence threshold specifically for small circles (< 16 px radius), since small spurious circles are the main source of noise in this kind of accumulator

### 2. Duplicate removal (over-segmentation)
Circles are sorted by descending accumulator confidence; each kept circle suppresses any other circle whose center is within 12 px **and** whose radius differs by less than 6 px. This is a simple greedy non-maximum suppression adapted to circular detections rather than bounding boxes.

### 3. Porosity estimation
Rather than summing individual circle areas (which double-counts overlapping regions), a binary mask is built by rasterizing every retained circle, and porosity is computed as the ratio of filled mask area to total slice area — robust to partial overlap between neighboring pores.

## Output

For each slice, the pipeline reports: number of detected pores, min/max/mean/std radius (px) and diameter (mm, using the given pixel size of 0.19 mm/px), and estimated surface porosity (%). Results are exported to `results_td4_per_image.csv`.

## ⚠ Missing input data

This notebook expects a sibling `../input/` folder containing the `.tif` CT-scan volumes, which is **not included** in this repository (the original dataset belongs to the course materials, not to this code). The notebook is provided translated and documented for portfolio/reference purposes; to actually run it, place the `.tif` volumes in an `input/` folder one level above this notebook.

```
computer-vision-repo/
├── input/                              ← place the .tif volumes here
│   ├── scan_00.tif
│   └── ...
└── hough-circle-pore-segmentation/
    └── Hough_Circle_Pore_Segmentation.ipynb
```

## How to run (once the data is in place)

```bash
pip install numpy pandas matplotlib scikit-image
jupyter notebook Hough_Circle_Pore_Segmentation.ipynb
```

## Stack

```
Python 3.x · NumPy · Pandas · Matplotlib · scikit-image (Canny, Hough transform)
```

## License

MIT — see the root [LICENSE](../LICENSE).

# 3D Point Cloud Registration with ICP

> Implementing and benchmarking the **Iterative Closest Point** algorithm from scratch — then comparing it against Open3D's optimized variants.

---

## What this is about

This notebook walks through the full pipeline of **3D rigid registration**: given two point clouds that represent the same surface but shifted in space, how do you automatically find the rotation and translation that aligns them?

The answer is **ICP (Iterative Closest Point)** — a classic algorithm in computer vision and robotics, used everywhere from autonomous driving (LiDAR odometry) to medical imaging (organ alignment) to 3D reconstruction.

---

## What's covered

### Part 1 — ICP from scratch
- Generating synthetic 3D point clouds from a $Z = \sin(X)\cos(Y)$ surface
- Applying a known rigid transformation (ground truth R, T)
- Implementing ICP step by step:
  - Nearest-neighbor correspondence search
  - Optimal rotation via **SVD** of the cross-covariance matrix
  - Iterative pose update
- Analyzing convergence, residual error, and per-iteration timing

### Part 2 — Robustness to noise
- Adding Gaussian noise ($\sigma \in [0, 0.05]$) to both point clouds
- Studying how noise level affects residual error, convergence speed, and computation time

### Part 3 — Open3D variants
| Method | Neighbor search | Complexity | RMSE |
|---|---|---|---|
| ICP from scratch | Brute force | $O(n^2)$ | 0.1358 |
| Open3D point-to-point | KD-Tree | $O(n \log n)$ | 0.1358 |
| **GICP** (Generalized ICP) | KD-Tree + normals | $O(n \log n)$ | **0.0000** |

GICP achieves **perfect alignment** where classic ICP gets stuck in a local minimum — by leveraging local surface covariance (Mahalanobis distance) instead of raw point-to-point distances.

---

## Key results

- Rotation **R** is always recovered perfectly (Frobenius error = 0.000000) thanks to SVD
- Translation **T** has a residual error when ICP doesn't fully converge (local optimizer limitation)
- Open3D's KD-Tree is **~60× faster** than the brute-force implementation with identical results
- GICP solves the local minimum problem by modeling surface geometry through normal estimation

---

## Stack

```
Python 3.x · NumPy · Matplotlib · Plotly · Open3D
```

Install dependencies:
```bash
pip install numpy matplotlib plotly open3d
```

---

## How to run

```bash
git clone https://github.com/devkumar-projects/icp-pointcloud-registration.git
cd icp-pointcloud-registration
jupyter notebook ICP_3D_PointCloud_Registration.ipynb
```

---

## Background

This project was completed as part of a **Mechatronics engineering** program (PA3 level), in an Image Processing course unit focused on 3D data processing. The mathematical foundations (SVD-based rigid registration, GICP covariance model) follow the standard literature on point cloud alignment.

---

*Author — Dev KUMAR · Mechatronics PA3*

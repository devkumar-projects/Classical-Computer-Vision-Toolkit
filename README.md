# Classical Computer Vision Toolkit

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-KLT%20%7C%20PnP-5C3EE8?logo=opencv&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-Hough%20Transform-orange)
![Open3D](https://img.shields.io/badge/Open3D-ICP%20%7C%20GICP-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Three classical (pre-deep-learning) computer vision problems, each solved with the geometric and signal-processing tools that are still the right tool for the job when you need precision, interpretability, and no training data: **recovering 3D structure and motion from images without a neural network in sight.**

| Project | Problem | Core techniques |
|---|---|---|
| [`klt-tracking-pnp-pose-estimation/`](klt-tracking-pnp-pose-estimation) | Track a rigid object in video and recover its full 6-DoF pose in real time | KLT optical flow, forward-backward validation, PnP, dynamic outlier exclusion |
| [`3d-point-cloud-registration-icp/`](3d-point-cloud-registration-icp) | Align two 3D point clouds of the same surface, shifted in space | ICP from scratch (SVD), KD-Tree acceleration, GICP |
| [`hough-circle-pore-segmentation/`](hough-circle-pore-segmentation) | Detect and quantify circular pores in CT-scan slices | Constrained circular Hough transform, non-max suppression, mask-based porosity |

## The common thread

All three projects tackle the same underlying question from a different angle: **how do you go from raw pixels to precise, quantitative geometric measurements** — a 6-DoF pose, a rigid transformation, a set of calibrated circle radii — **using explicit, interpretable models of image formation and geometry**, rather than a black-box learned representation?

- **KLT + PnP** recovers *where a known 3D object is*, frame by frame, from 2D-3D correspondences and a calibrated camera model.
- **ICP/GICP** recovers *how two 3D point sets relate to each other*, when correspondences aren't known and must be estimated iteratively.
- **Hough transform** recovers *what circular structures exist* in a noisy 2D image, using a parametric voting scheme instead of a per-pixel classifier.

Put together, they cover the three classical building blocks of geometric computer vision: **tracking + pose estimation**, **registration/alignment**, and **parametric shape detection** — each with its own failure modes (occlusion and drift for KLT, local minima for ICP, over-segmentation for Hough) and the classical techniques used to handle them.

## Notes on this repository

- **Language**: all notebooks, scripts, and READMEs are in English. The KLT+PnP script (`KLT_PNP.py`) was already written in English; the ICP notebook was already in English; the Hough-transform notebook was translated from the original French coursework version.
- **GitHub notebook rendering**: the ICP notebook originally used Plotly for interactive 3D plots, which GitHub's notebook viewer cannot render (it doesn't execute JavaScript). All 3D visualizations were converted to static Matplotlib figures and the notebook was re-executed, so the plots now display directly on GitHub. See that project's README for details, and for how to get interactivity back locally.
- **Missing external data**: the Hough-transform notebook expects CT-scan `.tif` volumes that belong to the original course materials and are not redistributed here — see that project's README for the expected folder layout.

## Interactive notebooks on Kaggle

- [3D Point Cloud Registration with ICP](https://www.kaggle.com/code/devkumarkaggle/3d-point-cloud-registration-with-icp) — registration and noise experiments on synthetic point clouds.
- [Detecting Synthetic Pores with Circular Hough](https://www.kaggle.com/code/devkumarkaggle/detecting-synthetic-pores-with-circular-hough) — an illustrated synthetic 2D circle-detection example.

## Related projects

- [UR7e Reinforcement Learning Suite](https://github.com/devkumar-projects/UR7e-Reinforcement-Learning-Suite) — robot tracking with optical flow and state estimation.
- [TurtleBot3 ROS 2 Navigation](https://github.com/devkumar-projects/TurtleBot3-ROS2-Navigation-Challenges) — mobile robot navigation with camera, LiDAR and odometry in simulation.
- [UR5e RL Pick-and-Place](https://github.com/devkumar-projects/ur5e-ReinforcementLearning-pick-and-place) — reinforcement learning for robot manipulation in simulation.

## License

MIT — see [LICENSE](LICENSE).

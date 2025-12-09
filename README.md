# Interactive Edge Detection Benchmark (Jupyter)

This repository contains the development of an **interactive Jupyter Notebook–based framework** for the comparison and evaluation of **edge detection algorithms**, including both classical and deep learning–based approaches.

The main goal is to provide a **modular and easily extensible environment** for:
- experimentation
- qualitative comparison
- robustness analysis under noise

---

## Project Objectives

- Loading of **single images** or **standard datasets** (BSDS500)
- Dynamic selection of the edge detection algorithm
- Visual comparison through an interactive interface
- Support for:
  - classical methods (Sobel, LoG, Canny)
  - deep learning methods (TEED, DexiNed) with pre-trained weights


## Current Status

At the current stage, the project focuses on:

- Interactive interface (`ipywidgets`)
- nput selection:
  - single image
  - dataset (BSDS500)
-Initial integration of:
  - **Canny**, implemented from scratch following the theoretical formulation of the algorithm
  - **TEED**, executed via command-line inference in an Anaconda environment


## Technologies Used

- Python
- Jupyter Notebook
- OpenCV / NumPy
- ipywidgets
- subprocess (for external deep learning models)
- PyTorch (for TEED / DexiNed)

---

## Project Structure

```text
.
├── notebooks/
│   └── interactive_edge_detection.ipynb
├── algorithms/
│   ├── canny.py
│   ├── sobel.py
│   ├── log.py
│   └── wrappers_dl.py
├── datasets/
│   └── BSDS500/
├── models/
│   ├── teed/
│   └── dexined/
├── utils/
│   ├── io_img.py
│   ├── noise.py
│   └── visualization.py
└── README.md

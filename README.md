# Interactive Edge Detection Benchmark

This repository contains the development of an **interactive edge detection benchmarking framework**, designed to compare and analyze **classical** and **deep learning–based** edge detection algorithms.

The project focuses on **visual comparison**, **robustness analysis**, and **modular experimentation**, providing both a Jupyter-based workflow and a standalone Python UI.

---

##  Project Goals

* Load and process:

  * **single images**
  * **standard datasets** (e.g. BSDS500)
* Dynamically select and run different edge detection algorithms
* Visual comparison of input vs output
* Analyze robustness to noise and parameter variations
* Provide a modular and extensible experimental framework

---

##  Supported Methods

### Classical Edge Detectors
* **Canny (custom implementation)**

### Deep Learning–Based Edge Detectors

* **TEED** (Tiny and Efficient Edge Detector)
* **DexiNed** (Dense Extreme Inception Network)

---

##  Canny Edge Detector (Custom Implementation)

The **Canny edge detector** included in this project is **implemented from scratch**,
**without using external edge-detection libraries**, strictly following the **theoretical formulation** and the classical steps of the algorithm:

1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

This allows:

* full control over each stage
* educational inspection of the algorithm
* fair comparison with learning-based approaches

---

##  Deep Learning Models

### TEED – Tiny and Efficient Edge Detector

TEED is a **lightweight convolutional neural network** for edge detection, designed to achieve a strong balance between **accuracy and computational efficiency**.

**Key characteristics:**

* Compact CNN architecture
* Multi-scale feature extraction
* Supervised training
* Robust to noise and texture variations

TEED is trained on the **BIPED dataset (Barcelona Images for Perceptual Edge Detection)**,
a dataset specifically designed for **perceptual edge detection** in urban scenes.

Unlike classical methods such as Canny, TEED does **not rely on manual thresholds**, but learns directly from data what constitutes a meaningful edge.

### Model Evaluation with TEED on the BSDS500 Dataset

For evaluation, I tested the TEED model on the **BSDS500 dataset**, one of the most widely used benchmarks in edge detection research. BSDS500 contains 500 natural images with human-annotated ground truth edge maps and is commonly used to compare the performance of different edge detection algorithms. The dataset is publicly available at: **[https://www.kaggle.com/datasets/balraj98/bsds500](https://www.kaggle.com/datasets/balraj98/bsds500)**

The model was applied to the BSDS500 test set to analyze its generalization ability and to compare its output with classical methods and other deep learning approaches. The results demonstrate that TEED is capable of producing precise and visually coherent edge maps, even when evaluated on data different from the one used for training.

---

### DexiNed – Dense Extreme Inception Network

DexiNed is a **deeper and more expressive edge detection network**, designed to capture edges at **multiple semantic levels**.

**How it works:**

* Combines **Dense blocks** and **Inception-style modules**
* Produces multiple **side-output edge maps**
* Fuses these maps into a final, refined edge representation

**Key characteristics:**

* Multi-scale and multi-level edge detection
* Strong generalization capability
* Particularly effective in complex scenes

DexiNed is also trained on the **BIPED dataset**, making it well-suited for real-world edge detection tasks.

### Model Training and Evaluation on the BIPED Dataset

For training the **DexiNed model**, I used the **BIPED (Barcelona Images for Perceptual Edge Detection)** dataset, which is specifically designed for benchmarking edge detection algorithms. BIPED consists of high-resolution outdoor images that have been carefully annotated at the edge level by experts, making it well-suited for training convolutional neural networks for edge detection tasks. In the dataset, 200 images are typically used for training and 50 images for testing, and it has been widely adopted in edge detection research. ([HyperAI][1])

You can download the BIPED dataset from Kaggle here:
**[https://www.kaggle.com/datasets/xavysp/biped](https://www.kaggle.com/datasets/xavysp/biped)** *(or search “BIPED edge detection dataset” on Kaggle)*.

I trained DexiNed on this dataset and also evaluated its performance on the test split to visually demonstrate its capability to accurately detect edges in complex natural scenes. The results shown in the demo video and screenshots are based on these trained and tested models.

---

## 🖥 Interactive Usage

The project supports:

* **Jupyter Notebook** interaction (via `ipywidgets`)
* **Standalone Python UI** (Tkinter + Matplotlib) with:

  * algorithm selection
  * parameter tuning (for Canny)
  * dataset-based inference (for TEED / DexiNed)
  * navigation through results

---

##  Notes

* Deep learning models are executed using their **original implementations and pretrained weights**
* The framework focuses on **qualitative comparison**, extensibility, and clarity
* Designed for experimentation, benchmarking, and educational purposes


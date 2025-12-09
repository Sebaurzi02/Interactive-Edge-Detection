import cv2
import os
from pathlib import Path
from typing import List, Dict
import numpy as np

"""SINGOLA IMG"""
def load_single_image(path: str, grayscale: bool = True) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Immagine non trovata: {path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)

    if img is None:
        raise ValueError(f"Errore nel caricamento dell'immagine: {path}")

    return img

"""DATASET"""
def load_bsds500(dataset_root: str,split: str = "test",max_images: int | None = None,grayscale: bool = True) -> List[Dict]:
    img_dir = (
        Path(dataset_root)
        / "data"
        / "images"
        / split
    )

    if not img_dir.exists():
        raise FileNotFoundError(f"dataset non trovato: {img_dir}")

    image_files = sorted(img_dir.glob("*.jpg"))

    if max_images is not None:
        image_files = image_files[:max_images]

    dataset = []

    for img_path in image_files:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(img_path), flag)

        if img is None:
            continue

        dataset.append({
            "name": img_path.stem,
            "image": img,
            "path": str(img_path)
        })

    return dataset


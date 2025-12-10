import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

from utils.io_img import load_single_image, load_bsds500
from alg.Canny import canny_pip


def test_single_image():
    print("Testing single image loading...")

    img_path = "data/Test.png"  # cambia se serve
    img = load_single_image(img_path)

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")

    plt.figure(figsize=(5, 5))
    plt.title("Single Image Test")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
    return img

def run_teed():
    project_root = Path(__file__).resolve().parent
    teed_dir = project_root / "models" / "TEED"

    cmd = [
        "python",
        "main.py",
        "--choose_test_data=-1"
    ]

    print("Running:", " ".join(cmd))

    subprocess.run(
        cmd,
        cwd=teed_dir,
        check=True
    )


def test_dataset():
    print("Testing dataset loading...")

    dataset_root = "data/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500/versions/1/images/test"
    dataset = load_bsds500(
        dataset_root=dataset_root,
        split="test",
        max_images=5
    )

    print(f"Loaded {len(dataset)} images")

    for sample in dataset:
        """
        print(f" - {sample['name']} | shape: {sample['image'].shape}")

        plt.figure(figsize=(4, 4))
        plt.title(sample["name"])
        plt.imshow(sample["image"], cmap="gray")
        plt.axis("off")
        plt.show()
        """
        canny_edge_detector(sample["image"])

def canny_edge_detector(img, low=10, high=100, sigma=0.5, T=0.3):
    return canny_pip(img, low, high, sigma, T)

if __name__ == "__main__":
    """img=test_single_image()
    canny_edge_detector(img)
    test_dataset()"""
    run_teed()

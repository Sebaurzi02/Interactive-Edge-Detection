import sys
from pathlib import Path

# aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import shutil
from PIL import Image, ImageTk
import numpy as np

from alg.Canny import canny_pip
from utils.io_img import load_single_image


class EdgeDetApp:
    def __init__(self, master):
        self.mode = "CLASSIC"

        self.teed_input_dir = project_root / "models" / "TEED" / "data"
        self.teed_output_dir = project_root / "models" / "TEED" / "result" / "BIPED2CLASSIC" / "fused"

        self.bsds_root = project_root / "data" / "BSDS500" / "images"
        self.bsds_splits = ["train", "val", "test"]


        self.teed_images = []
        self.teed_index = 0

        self.master = master
        master.title("Edge Detection App UI")

        self.master.geometry("900x700")
        self.current_img = None  # immagine caricata

        
        # FRAME CONTROLLI
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.LEFT, fill="y", padx=10, pady=10)

       
        # Bottone per caricare l'immagine
        tk.Button(control_frame, text="Load Image",command=self.load_image,width=20).pack(pady=5)

        # Select algoritmi
        tk.Label(control_frame, text="Select Algorithm:").pack()
        self.alg_var = tk.StringVar()
        self.alg_menu = ttk.Combobox(
            control_frame,
            textvariable=self.alg_var,
            values=["Canny", "Sobel", "LoG", "TEED", "DexiNed"]
        )
        self.alg_menu.pack(pady=5)
        self.alg_menu.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        # PARAMETRI PER CANNY [low,high,sigma,T] (compare solo se scelto)
        self.canny_frame = tk.Frame(control_frame)

        tk.Label(self.canny_frame, text="Low threshold:").grid(row=0, column=0, sticky="w")
        self.low_entry = tk.Entry(self.canny_frame)
        self.low_entry.insert(0, "20")
        self.low_entry.grid(row=0, column=1)

        tk.Label(self.canny_frame, text="High threshold:").grid(row=1, column=0, sticky="w")
        self.high_entry = tk.Entry(self.canny_frame)
        self.high_entry.insert(0, "50")
        self.high_entry.grid(row=1, column=1)

        tk.Label(self.canny_frame, text="Sigma:").grid(row=2, column=0, sticky="w")
        self.sigma_entry = tk.Entry(self.canny_frame)
        self.sigma_entry.insert(0, "1")
        self.sigma_entry.grid(row=2, column=1)

        tk.Label(self.canny_frame, text="T (hysteresis):").grid(row=3, column=0, sticky="w")
        self.T_entry = tk.Entry(self.canny_frame)
        self.T_entry.insert(0, "0.3")
        self.T_entry.grid(row=3, column=1)


        # Bottone esegui
        tk.Button(control_frame,text="Run Algorithm",command=self.run_algorithm,width=20).pack(pady=15)

        self.teed_dataset_frame = tk.Frame(control_frame)
        tk.Label(self.teed_dataset_frame, text="BSDS500 split:").pack()

        self.bsds_var = tk.StringVar(value="test")
        self.bsds_menu = ttk.Combobox(self.teed_dataset_frame,textvariable=self.bsds_var,values=self.bsds_splits,state="readonly",width=15)
        self.bsds_menu.pack(pady=3)

        tk.Button(self.teed_dataset_frame,text="Load BSDS500",command=self.load_bsds500_teed,width=20).pack(pady=5)

        # Grafico
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill="both", expand=True)
        self.current_img = None

        tk.Button(control_frame, text="◀ Prev", command=self.prev_teed).pack(pady=2)
        tk.Button(control_frame, text="Next ▶", command=self.next_teed).pack(pady=2)


    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        if self.mode == "CLASSIC":
            self.load_classic_image(file_path)

        elif self.mode == "TEED":
            self.load_teed_image(file_path)

    def load_classic_image(self, file_path):
        self.current_img = load_single_image(file_path)

        self.ax[0].clear()
        self.ax[0].imshow(self.current_img, cmap="gray")
        self.ax[0].set_title("Input")
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].axis("off")

        self.canvas.draw()

    def load_teed_image(self, file_path):
        self.teed_input_dir.mkdir(parents=True, exist_ok=True)

        dst = self.teed_input_dir / Path(file_path).name
        shutil.copy(file_path, dst)

        # lista input coerente
        self.teed_input_images = [dst]

        print("[TEED] Copied to:", dst)

    
    def load_bsds500_teed(self):
        split = self.bsds_var.get()
        src_dir = self.bsds_root / split

        if not src_dir.exists():
            print("[BSDS] Split not found:", src_dir)
            return

        self.teed_input_dir.mkdir(parents=True, exist_ok=True)

        # pulizia input TEED
        for f in self.teed_input_dir.glob("*"):
            f.unlink()

        images = sorted(
            list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        )

        if not images:
            print("[BSDS] No images found in", src_dir)
            return

        self.teed_input_images = []

        print(f"[BSDS] Copying {len(images)} images from {split}")

        for img_path in images:
            dst = self.teed_input_dir / img_path.name
            shutil.copy(img_path, dst)
            self.teed_input_images.append(dst)

        print("[BSDS] Copy completed, running TEED...")

        self.run_teed()
        self.load_teed_results()



    def clear_plots(self):
        for ax in self.ax:
            ax.clear()
            ax.axis("off")
        self.canvas.draw()

    # CAMBIO ALGORITMO → mostra/nasconde parametri Canny
    def on_algorithm_change(self, event=None):
        algo = self.alg_var.get()

        self.canny_frame.pack_forget()
        self.teed_dataset_frame.pack_forget()

        if algo == "Canny":
            self.mode = "CLASSIC"
            self.canny_frame.pack(pady=10)
        elif algo == "TEED":
            self.mode = "TEED"
            self.clear_plots()
            self.teed_dataset_frame.pack(pady=10)

    # ESEGUE CANNY CON PARAMETRI
    def run_canny(self):
        low = float(self.low_entry.get())
        high = float(self.high_entry.get())
        sigma = float(self.sigma_entry.get())
        T = float(self.T_entry.get())

        print(f"[CANNY] low={low}, high={high}, sigma={sigma}, T={T}")

        result = canny_pip(self.current_img, low, high, sigma, T)

        # mostra il risultato
        self.ax[1].clear()
        self.ax[1].imshow(result, cmap="gray")
        self.ax[1].set_title("Canny Output")
        self.ax[1].axis("off")

        self.canvas.draw()
    
    def show_teed_image(self):
        if not self.teed_images or not self.teed_input_images:
            return

        idx = min(self.teed_index, len(self.teed_input_images) - 1)

        input_path = self.teed_input_images[idx]
        output_path = self.teed_images[self.teed_index]

        input_img = load_single_image(str(input_path))
        output_img = load_single_image(str(output_path))

        self.ax[0].clear()
        self.ax[0].imshow(input_img, cmap="gray")
        self.ax[0].set_title("TEED Input")
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].imshow(output_img, cmap="gray")
        self.ax[1].set_title("TEED Output")
        self.ax[1].axis("off")

        self.canvas.draw()



    def load_teed_results(self):
        if not self.teed_output_dir.exists():
            print("[TEED] Output directory not found")
            return

        self.teed_images = sorted(self.teed_output_dir.glob("*.png"))
        self.teed_index = 0

        if not self.teed_images:
            print("[TEED] No output images found")
            return

        self.show_teed_image()

    def run_teed(self):
        # path alla root del progetto
        project_root = Path(__file__).resolve().parent.parent

        # path a models/TEED
        teed_dir = project_root / "models" / "TEED"

        if not teed_dir.exists():
            print("[ERROR] TEED directory not found:", teed_dir)
            return

        cmd = [
            sys.executable,   # usa lo stesso python dell'ambiente attivo
            "main.py",
            "--choose_test_data=-1"
        ]

        print("[TEED] Running:", " ".join(cmd))
        print("[TEED] Working dir:", teed_dir)

        try:
            subprocess.run(
                cmd,
                cwd=teed_dir,
                check=True
            )
            print("[TEED] Finished successfully")

        except subprocess.CalledProcessError as e:
            print("[TEED] Error during execution")
            print(e)
        
        if not self.teed_output_dir.exists():
            print("[TEED] Output directory not created yet:")
            print(self.teed_output_dir)
            return

        print("[TEED] Output directory found")
    
    # RUN ALGORITHM
    def run_algorithm(self):
        if self.current_img is None and self.mode == "CLASSIC":
            print("Load an image first.")
            return

        algo = self.alg_var.get()

        if algo == "Canny":
            self.run_canny()
        elif algo == "TEED":
            self.run_teed()
            self.load_teed_results()
        else:
            print(f"Algorithm {algo} not implemented yet.")

    def next_teed(self):
        if self.teed_images:
            self.teed_index = (self.teed_index + 1) % len(self.teed_images)
            self.show_teed_image()

    def prev_teed(self):
        if self.teed_images:
            self.teed_index = (self.teed_index - 1) % len(self.teed_images)
            self.show_teed_image()

root = tk.Tk()
app = EdgeDetApp(root)
root.mainloop()

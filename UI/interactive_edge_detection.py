import sys
from pathlib import Path
import subprocess
import shutil
import threading

# aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from alg.Canny import canny_pip
from utils.io_img import load_single_image


class EdgeDetApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Edge Detection Studio")
        self.master.geometry("1280x780")
        self.master.minsize(1100, 700)
        self.master.configure(bg="#ececec")

        # -------------------------
        # Stato app
        # -------------------------
        self.mode = "CLASSIC"
        self.current_img = None
        self.processing = False

        # -------------------------
        # Path progetto
        # -------------------------
        # TEED
        self.teed_input_dir = project_root / "models" / "TEED" / "data"
        self.teed_output_dir = project_root / "models" / "TEED" / "result" / "BIPED2CLASSIC" / "fused"

        # BSDS500
        self.bsds_root = project_root / "data" / "BSDS500" / "images"
        self.bsds_splits = ["train", "val", "test"]

        # DexiNed
        self.dexined_dir = project_root / "models" / "DexiNed-master"
        self.dexined_fused_dir = self.dexined_dir / "result" / "BIPED2BIPED" / "fused"
        self.dexined_avg_dir = self.dexined_dir / "result" / "BIPED2BIPED" / "avg"

        # -------------------------
        # Dati immagini / navigazione
        # -------------------------
        self.teed_input_images = []
        self.teed_images = []
        self.teed_index = 0

        self.dexined_input_images = []
        self.dexined_images = []
        self.dexined_index = 0
        self.dexined_mode = "fused"

        # -------------------------
        # Stile ttk
        # -------------------------
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("TLabel", background="#ececec", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", background="#ececec", font=("Segoe UI", 20, "bold"))
        self.style.configure("Section.TLabel", background="#f7f7f7", font=("Segoe UI", 11, "bold"))
        self.style.configure("Card.TFrame", background="#f7f7f7", relief="flat")
        self.style.configure("Sidebar.TFrame", background="#e4e4e4")
        self.style.configure("Main.TFrame", background="#ececec")
        self.style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        self.style.configure("TCombobox", padding=4)

        self.build_ui()
        self.set_status("Ready")

    # =========================================================
    # UI
    # =========================================================
    def build_ui(self):
        # Root layout
        self.root_container = ttk.Frame(self.master, style="Main.TFrame")
        self.root_container.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = ttk.Frame(self.root_container, width=300, style="Sidebar.TFrame")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Main area
        self.main_area = ttk.Frame(self.root_container, style="Main.TFrame")
        self.main_area.pack(side="right", fill="both", expand=True, padx=12, pady=12)

        self.build_sidebar()
        self.build_main_area()
        self.build_statusbar()

        # CHIAMALA QUI, non dentro build_sidebar
        self.on_algorithm_change()

    def build_sidebar(self):
        header = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        header.pack(fill="x", padx=14, pady=(16, 10))

        ttk.Label(header, text="Edge Detection Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Canny, TEED e DexiNed in un'unica interfaccia",
            background="#e4e4e4",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))

        # Card: input
        self.input_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.input_card.pack(fill="x", padx=14, pady=8)

        ttk.Label(self.input_card, text="Input", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 6))

        ttk.Button(self.input_card, text="Load Image", command=self.load_image).pack(fill="x", padx=12, pady=(0, 10))

        ttk.Label(self.input_card, text="Select Algorithm:").pack(anchor="w", padx=12)
        self.alg_var = tk.StringVar(value="Canny")
        self.alg_menu = ttk.Combobox(
            self.input_card,
            textvariable=self.alg_var,
            values=["Canny", "TEED", "DexiNed"],
            state="readonly"
        )
        self.alg_menu.pack(fill="x", padx=12, pady=(4, 10))
        self.alg_menu.bind("<<ComboboxSelected>>", self.on_algorithm_change)

        ttk.Button(self.input_card, text="Run Algorithm", command=self.run_algorithm, style="Accent.TButton").pack(
            fill="x", padx=12, pady=(0, 12)
        )

        # Card: Canny params
        self.canny_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.canny_card.pack(fill="x", padx=14, pady=8)

        ttk.Label(self.canny_card, text="Canny Parameters", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        self.canny_frame = ttk.Frame(self.canny_card, style="Card.TFrame")
        self.canny_frame.pack(fill="x", padx=12, pady=(0, 12))

        self._make_labeled_entry(self.canny_frame, "Low threshold:", "20", 0, "low_entry")
        self._make_labeled_entry(self.canny_frame, "High threshold:", "50", 1, "high_entry")
        self._make_labeled_entry(self.canny_frame, "Sigma:", "1", 2, "sigma_entry")
        self._make_labeled_entry(self.canny_frame, "T (hysteresis):", "0.3", 3, "T_entry")

        # Card: TEED dataset
        self.teed_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.teed_card.pack(fill="x", padx=14, pady=8)

        ttk.Label(self.teed_card, text="TEED Dataset", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        self.teed_dataset_frame = ttk.Frame(self.teed_card, style="Card.TFrame")
        self.teed_dataset_frame.pack(fill="x", padx=12, pady=(0, 12))

        ttk.Label(self.teed_dataset_frame, text="BSDS500 split:").pack(anchor="w")
        self.bsds_var = tk.StringVar(value="test")
        self.bsds_menu = ttk.Combobox(
            self.teed_dataset_frame,
            textvariable=self.bsds_var,
            values=self.bsds_splits,
            state="readonly"
        )
        self.bsds_menu.pack(fill="x", pady=(4, 8))

        ttk.Button(self.teed_dataset_frame, text="Load BSDS500", command=self.load_bsds500_teed).pack(fill="x")

        # Card: DexiNed options
        self.dexined_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.dexined_card.pack(fill="x", padx=14, pady=8)

        ttk.Label(self.dexined_card, text="DexiNed Options", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        self.dexined_frame = ttk.Frame(self.dexined_card, style="Card.TFrame")
        self.dexined_frame.pack(fill="x", padx=12, pady=(0, 12))

        ttk.Label(self.dexined_frame, text="DexiNed Output Type:").pack(anchor="w")
        self.dexined_var = tk.StringVar(value="fused")
        self.dexined_menu = ttk.Combobox(
            self.dexined_frame,
            textvariable=self.dexined_var,
            values=["fused", "avg"],
            state="readonly"
        )
        self.dexined_menu.pack(fill="x", pady=(4, 8))
        self.dexined_menu.bind("<<ComboboxSelected>>", lambda e: self.load_dexined_results())

        # Card: navigation
        self.nav_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.nav_card.pack(fill="x", padx=14, pady=8)

        ttk.Label(self.nav_card, text="Navigation", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        nav_btn_frame = ttk.Frame(self.nav_card, style="Card.TFrame")
        nav_btn_frame.pack(fill="x", padx=12)

        ttk.Button(nav_btn_frame, text="◀ Prev", command=self.prev_image).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(nav_btn_frame, text="Next ▶", command=self.next_image).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.image_counter_var = tk.StringVar(value="Image: 0 / 0")
        ttk.Label(self.nav_card, textvariable=self.image_counter_var).pack(anchor="w", padx=12, pady=(8, 12))

        # Card: log
        self.log_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        self.log_card.pack(fill="both", expand=True, padx=14, pady=(8, 14))

        ttk.Label(self.log_card, text="Log", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        self.log_text = tk.Text(
            self.log_card,
            height=12,
            wrap="word",
            bg="#ffffff",
            fg="#222222",
            font=("Consolas", 9),
            relief="flat",
            borderwidth=1
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_text.config(state="disabled")

    def build_main_area(self):
        topbar = ttk.Frame(self.main_area, style="Main.TFrame")
        topbar.pack(fill="x", pady=(0, 10))

        self.viewer_title_var = tk.StringVar(value="Visualization")
        ttk.Label(topbar, textvariable=self.viewer_title_var, font=("Segoe UI", 14, "bold"), background="#ececec").pack(
            side="left", anchor="w"
        )

        self.viewer_subtitle_var = tk.StringVar(value="Load an image or a dataset to start")
        ttk.Label(topbar, textvariable=self.viewer_subtitle_var, font=("Segoe UI", 10), background="#ececec").pack(
            side="right", anchor="e"
        )

        plot_frame = tk.Frame(self.main_area, bg="#ffffff", bd=0, relief="flat")
        plot_frame.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor("#ffffff")

        for a in self.ax:
            a.set_facecolor("#ffffff")
            a.axis("off")

        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(
            self.master,
            textvariable=self.status_var,
            anchor="w",
            bd=1,
            relief="sunken",
            bg="#dddddd",
            fg="#222222",
            padx=8,
            pady=4,
            font=("Segoe UI", 9)
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _make_labeled_entry(self, parent, label, default, row, attr_name):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 6))
        entry = ttk.Entry(parent)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        parent.grid_columnconfigure(1, weight=1)
        setattr(self, attr_name, entry)

    # =========================================================
    # Utility UI
    # =========================================================
    def log(self, message):
        print(message)
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def set_status(self, message):
        self.status_var.set(message)
        self.master.update_idletasks()

    def set_viewer_info(self, title, subtitle=""):
        self.viewer_title_var.set(title)
        self.viewer_subtitle_var.set(subtitle)

    def update_image_counter(self, current=0, total=0):
        self.image_counter_var.set(f"Image: {current} / {total}")

    def clear_plots(self):
        for axis in self.ax:
            axis.clear()
            axis.axis("off")
        self.canvas.draw()
        self.update_image_counter(0, 0)

    def show_info(self, title, message):
        messagebox.showinfo(title, message)

    def show_error(self, title, message):
        messagebox.showerror(title, message)

    # =========================================================
    # Cambio algoritmo
    # =========================================================
    def on_algorithm_change(self, event=None):
        algo = self.alg_var.get()

        self.canny_card.pack_forget()
        self.teed_card.pack_forget()
        self.dexined_card.pack_forget()

        if algo == "Canny":
            self.mode = "CLASSIC"
            self.canny_card.pack(fill="x", padx=14, pady=8)
            self.set_viewer_info("Canny Visualization", "Load a single image and run edge detection")
            self.set_status("Canny mode selected")

        elif algo == "TEED":
            self.mode = "TEED"
            self.teed_card.pack(fill="x", padx=14, pady=8)
            self.clear_plots()
            self.set_viewer_info("TEED Visualization", "Load BSDS500 or a custom image")
            self.set_status("TEED mode selected")

        elif algo == "DexiNed":
            self.mode = "DEXINED"
            self.dexined_card.pack(fill="x", padx=14, pady=8)
            self.clear_plots()
            self.set_viewer_info("DexiNed Visualization", "Run inference and browse results")
            self.set_status("DexiNed mode selected")

    # =========================================================
    # Load input
    # =========================================================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not file_path:
            return

        try:
            if self.mode == "CLASSIC":
                self.load_classic_image(file_path)

            elif self.mode == "TEED":
                self.load_teed_image(file_path)
                self.show_info("TEED", "Image copied into TEED input folder.\nNow click 'Run Algorithm'.")

            elif self.mode == "DEXINED":
                self.show_info("DexiNed", "Custom single-image loading is not yet wired to the current DexiNed pipeline.")

        except Exception as e:
            self.show_error("Load Error", str(e))
            self.log(f"[ERROR] Failed to load image: {e}")

    def load_classic_image(self, file_path):
        self.current_img = load_single_image(file_path)

        self.ax[0].clear()
        self.ax[0].imshow(self.current_img, cmap="gray")
        self.ax[0].set_title("Input Image", fontsize=13)
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].set_title("Output", fontsize=13)
        self.ax[1].axis("off")

        self.canvas.draw()
        self.set_status(f"Loaded image: {Path(file_path).name}")
        self.set_viewer_info("Canny Visualization", Path(file_path).name)
        self.update_image_counter(1, 1)
        self.log(f"[CLASSIC] Loaded image: {file_path}")

    def load_teed_image(self, file_path):
        self.teed_input_dir.mkdir(parents=True, exist_ok=True)

        # pulizia input per singola immagine
        for f in self.teed_input_dir.glob("*"):
            if f.is_file():
                f.unlink()

        dst = self.teed_input_dir / Path(file_path).name
        shutil.copy(file_path, dst)
        self.teed_input_images = [dst]

        self.log(f"[TEED] Copied input image to: {dst}")
        self.set_status(f"TEED input ready: {dst.name}")

        input_img = load_single_image(str(dst))
        self.ax[0].clear()
        self.ax[0].imshow(input_img, cmap="gray")
        self.ax[0].set_title("TEED Input", fontsize=13)
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].set_title("TEED Output", fontsize=13)
        self.ax[1].axis("off")
        self.canvas.draw()

        self.update_image_counter(1, 1)
        self.set_viewer_info("TEED Visualization", dst.name)

    def load_bsds500_teed(self):
        split = self.bsds_var.get()
        src_dir = self.bsds_root / split

        if not src_dir.exists():
            self.show_error("BSDS500 Error", f"Split not found:\n{src_dir}")
            self.log(f"[BSDS] Split not found: {src_dir}")
            return

        self.teed_input_dir.mkdir(parents=True, exist_ok=True)

        for f in self.teed_input_dir.glob("*"):
            if f.is_file():
                f.unlink()

        images = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))

        if not images:
            self.show_error("BSDS500 Error", f"No images found in:\n{src_dir}")
            self.log(f"[BSDS] No images found in {src_dir}")
            return

        self.teed_input_images = []

        self.log(f"[BSDS] Copying {len(images)} images from split: {split}")
        self.set_status(f"Copying {len(images)} BSDS500 images...")

        for img_path in images:
            dst = self.teed_input_dir / img_path.name
            shutil.copy(img_path, dst)
            self.teed_input_images.append(dst)

        self.log("[BSDS] Copy completed")
        self.set_status(f"BSDS500 {split} loaded ({len(images)} images)")
        self.set_viewer_info("TEED Visualization", f"BSDS500 split: {split}")
        self.update_image_counter(0, len(self.teed_input_images))
        self.show_info("BSDS500", f"{len(images)} images copied into TEED input folder.")

    # =========================================================
    # Canny
    # =========================================================
    def run_canny(self):
        if self.current_img is None:
            self.show_error("Canny Error", "Load an image first.")
            return

        try:
            low = float(self.low_entry.get())
            high = float(self.high_entry.get())
            sigma = float(self.sigma_entry.get())
            T = float(self.T_entry.get())
        except ValueError:
            self.show_error("Invalid Parameters", "Canny parameters must be numeric.")
            return

        try:
            self.set_status("Running Canny...")
            self.log(f"[CANNY] low={low}, high={high}, sigma={sigma}, T={T}")

            result = canny_pip(self.current_img, low, high, sigma, T)
            result = np.asarray(result, dtype=np.float32)

            self.ax[1].clear()
            self.ax[1].imshow(result, cmap="gray")
            self.ax[1].set_title("Canny Output", fontsize=13)
            self.ax[1].axis("off")

            self.canvas.draw()
            self.set_status("Canny completed")
            self.log("[CANNY] Completed successfully")

        except Exception as e:
            self.show_error("Canny Error", str(e))
            self.log(f"[CANNY] Error: {e}")
            self.set_status("Canny failed")

    # =========================================================
    # Visualizzazione TEED / DexiNed
    # =========================================================
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
        self.ax[0].set_title("TEED Input", fontsize=13)
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].imshow(output_img, cmap="gray")
        self.ax[1].set_title("TEED Output", fontsize=13)
        self.ax[1].axis("off")

        self.canvas.draw()

        self.update_image_counter(self.teed_index + 1, len(self.teed_images))
        self.set_status(f"Showing TEED result {self.teed_index + 1}/{len(self.teed_images)}")
        self.set_viewer_info("TEED Visualization", output_path.name)

    def show_dexined_image(self):
        if not self.dexined_images:
            return

        idx = self.dexined_index
        output_path = self.dexined_images[idx]

        output_img = load_single_image(str(output_path))

        if self.dexined_input_images and idx < len(self.dexined_input_images):
            input_path = self.dexined_input_images[idx]
            input_img = load_single_image(str(input_path))
        else:
            input_img = np.zeros_like(output_img)

        self.ax[0].clear()
        self.ax[0].imshow(input_img, cmap="gray")
        self.ax[0].set_title("DexiNed Input", fontsize=13)
        self.ax[0].axis("off")

        self.ax[1].clear()
        self.ax[1].imshow(output_img, cmap="gray")
        self.ax[1].set_title(f"DexiNed Output ({self.dexined_var.get()})", fontsize=13)
        self.ax[1].axis("off")

        self.canvas.draw()

        self.update_image_counter(self.dexined_index + 1, len(self.dexined_images))
        self.set_status(f"Showing DexiNed result {self.dexined_index + 1}/{len(self.dexined_images)}")
        self.set_viewer_info("DexiNed Visualization", output_path.name)

    # =========================================================
    # Load risultati
    # =========================================================
    def load_teed_results(self):
        teed_result_root = project_root / "models" / "TEED" / "result"

        if not teed_result_root.exists():
            self.log(f"[TEED] Output root not found: {teed_result_root}")
            self.show_error("TEED Error", "TEED result folder not found.")
            return

        candidates = (
            list(teed_result_root.rglob("*.png")) +
            list(teed_result_root.rglob("*.jpg")) +
            list(teed_result_root.rglob("*.jpeg")) +
            list(teed_result_root.rglob("*.bmp"))
        )

        if not candidates:
            self.log(f"[TEED] No output images found under: {teed_result_root}")
            self.show_error("TEED Error", "No output images found in TEED result folders.")
            return

        self.teed_images = sorted(candidates)
        self.teed_index = 0

        self.log(f"[TEED] Loaded {len(self.teed_images)} result images")
        self.show_teed_image()

    def load_dexined_results(self):
        selected = self.dexined_var.get()

        if selected == "fused":
            output_dir = self.dexined_fused_dir
        else:
            output_dir = self.dexined_avg_dir

        if not output_dir.exists():
            self.log(f"[DexiNed] Output directory not found: {output_dir}")
            self.show_error("DexiNed Error", f"Output directory not found:\n{output_dir}")
            return

        self.dexined_images = sorted(
            list(output_dir.glob("*.png")) +
            list(output_dir.glob("*.jpg")) +
            list(output_dir.glob("*.jpeg")) +
            list(output_dir.glob("*.bmp"))
        )
        self.dexined_index = 0

        if not self.dexined_images:
            self.log(f"[DexiNed] No output images found in: {output_dir}")
            self.show_error("DexiNed Error", f"No output images found in:\n{output_dir}")
            return

        self.log(f"[DexiNed] Loaded {len(self.dexined_images)} images from {selected}")
        self.show_dexined_image()

    # =========================================================
    # Esecuzione modelli
    # =========================================================
    def run_teed(self):
        teed_dir = project_root / "models" / "TEED"

        if not teed_dir.exists():
            self.log(f"[ERROR] TEED directory not found: {teed_dir}")
            self.show_error("TEED Error", f"TEED directory not found:\n{teed_dir}")
            return False

        cmd = [
            sys.executable,
            "main.py",
            "--choose_test_data=-1"
        ]

        self.log("[TEED] Running: " + " ".join(map(str, cmd)))
        self.log(f"[TEED] Working dir: {teed_dir}")
        self.set_status("Running TEED...")

        try:
            result = subprocess.run(
                cmd,
                cwd=teed_dir,
                check=True,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                self.log("[TEED][STDOUT]")
                self.log(result.stdout)

            if result.stderr.strip():
                self.log("[TEED][STDERR]")
                self.log(result.stderr)

            self.log("[TEED] Finished successfully")
            self.set_status("TEED completed")
            return True

        except subprocess.CalledProcessError as e:
            self.log("[TEED] Error during execution")
            self.log(f"Return code: {e.returncode}")
            if e.stdout:
                self.log("[TEED][STDOUT]")
                self.log(e.stdout)
            if e.stderr:
                self.log("[TEED][STDERR]")
                self.log(e.stderr)

            self.set_status("TEED failed")
            self.show_error("TEED Execution Error", "TEED failed.\nCheck the log panel for details.")
            return False

        except Exception as e:
            self.log(f"[TEED] Unexpected error: {e}")
            self.set_status("TEED failed")
            self.show_error("TEED Error", str(e))
            return False

    def run_dexined(self):
        if not self.dexined_dir.exists():
            self.log(f"[ERROR] DexiNed directory not found: {self.dexined_dir}")
            self.show_error("DexiNed Error", f"DexiNed directory not found:\n{self.dexined_dir}")
            return False

        cmd = [
            sys.executable,
            "main.py",
            "--choose_test_data",
            "0"
        ]

        self.log("[DexiNed] Running: " + " ".join(map(str, cmd)))
        self.log(f"[DexiNed] Working dir: {self.dexined_dir}")
        self.set_status("Running DexiNed...")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.dexined_dir,
                check=True,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                self.log("[DexiNed][STDOUT]")
                self.log(result.stdout)

            if result.stderr.strip():
                self.log("[DexiNed][STDERR]")
                self.log(result.stderr)

            self.log("[DexiNed] Finished successfully")
            self.set_status("DexiNed completed")
            return True

        except subprocess.CalledProcessError as e:
            self.log("[DexiNed] Error during execution")
            self.log(f"Return code: {e.returncode}")
            if e.stdout:
                self.log("[DexiNed][STDOUT]")
                self.log(e.stdout)
            if e.stderr:
                self.log("[DexiNed][STDERR]")
                self.log(e.stderr)

            self.set_status("DexiNed failed")
            self.show_error("DexiNed Execution Error", "DexiNed failed.\nCheck the log panel for details.")
            return False

        except Exception as e:
            self.log(f"[DexiNed] Unexpected error: {e}")
            self.set_status("DexiNed failed")
            self.show_error("DexiNed Error", str(e))
            return False

    # =========================================================
    # Threading
    # =========================================================
    def run_algorithm(self):
        if self.processing:
            self.show_info("Busy", "A process is already running.")
            return

        algo = self.alg_var.get()

        if algo == "Canny":
            self.run_canny()
        elif algo == "TEED":
            self.start_background_task(self._run_teed_pipeline)
        elif algo == "DexiNed":
            self.start_background_task(self._run_dexined_pipeline)
        else:
            self.show_error("Error", f"Algorithm {algo} not implemented.")

    def start_background_task(self, target):
        self.processing = True
        self.set_status("Processing...")
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def _run_teed_pipeline(self):
        ok = self.run_teed()
        self.master.after(0, lambda: self._finish_teed_pipeline(ok))

    def _finish_teed_pipeline(self, ok):
        self.processing = False
        if ok:
            self.load_teed_results()

    def _run_dexined_pipeline(self):
        ok = self.run_dexined()
        self.master.after(0, lambda: self._finish_dexined_pipeline(ok))

    def _finish_dexined_pipeline(self, ok):
        self.processing = False
        if ok:
            self.load_dexined_results()

    # =========================================================
    # Navigazione immagini
    # =========================================================
    def next_image(self):
        if self.mode == "TEED" and self.teed_images:
            self.teed_index = (self.teed_index + 1) % len(self.teed_images)
            self.show_teed_image()

        elif self.mode == "DEXINED" and self.dexined_images:
            self.dexined_index = (self.dexined_index + 1) % len(self.dexined_images)
            self.show_dexined_image()

    def prev_image(self):
        if self.mode == "TEED" and self.teed_images:
            self.teed_index = (self.teed_index - 1) % len(self.teed_images)
            self.show_teed_image()

        elif self.mode == "DEXINED" and self.dexined_images:
            self.dexined_index = (self.dexined_index - 1) % len(self.dexined_images)
            self.show_dexined_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetApp(root)
    root.mainloop()
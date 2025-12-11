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
from PIL import Image, ImageTk
import numpy as np

from alg.Canny import canny_pip
from utils.io_img import load_single_image


class EdgeDetApp:
    def __init__(self, master):
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

        # Grafico
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill="both", expand=True)
        self.current_img = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.current_img = load_single_image(file_path)

        self.ax[0].clear()
        self.ax[0].imshow(self.current_img, cmap="gray")
        self.ax[0].set_title("Input")
        self.ax[0].axis("off")
        self.canvas.draw()

    # CAMBIO ALGORITMO → mostra/nasconde parametri Canny
    def on_algorithm_change(self, event=None):
        algo = self.alg_var.get()

        self.canny_frame.pack_forget()

        if algo == "Canny":
            self.canny_frame.pack(pady=10)


    # RUN ALGORITHM
    def run_algorithm(self):
        if self.current_img is None:
            print("Load an image first.")
            return

        algo = self.alg_var.get()

        if algo == "Canny":
            self.run_canny()
        else:
            print(f"Algorithm {algo} not implemented yet.")

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


root = tk.Tk()
app = EdgeDetApp(root)
root.mainloop()

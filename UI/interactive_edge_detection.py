import sys
import os
from pathlib import Path

# aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from alg.Canny import canny_pip
from utils.io_img import load_single_image


class EdgeDetApp:
    def __init__(self, master):
        self.master = master
        master.title("Edge Detection App")

        self.master.geometry("1000x700")

        # Dropdown algoritmo
        self.algorithm_var = tk.StringVar(value="Canny")
        ttk.Label(master, text="Algorithm:").pack()
        ttk.OptionMenu(master, self.algorithm_var, "Canny", "Canny", "Sobel", "LoG", "TEED").pack()

        # Bottone carica immagine
        ttk.Button(master, text="Load Image", command=self.load_image).pack(pady=5)

        # Bottone esegui
        ttk.Button(master, text="Run", command=self.run_algorithm).pack(pady=5)

        # Area grafico
        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack()

        self.img = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        self.img = load_single_image(file_path)

        self.ax[0].imshow(self.img, cmap="gray")
        self.ax[0].set_title("Input")
        self.ax[0].axis("off")
        self.canvas.draw()

    def run_algorithm(self):
        if self.img is None:
            print("Load an image first")
            return

        algo = self.algorithm_var.get()

        if algo == "Canny":
            result = canny_pip(self.img, 20, 50, 1, 0.3)
        elif algo == "TEED":
            # da definire wrapper subprocess
            print("TEED not implemented here")
            return
        else:
            print("Algorithm not implemented")
            return

        self.ax[1].imshow(result, cmap="gray")
        self.ax[1].set_title(algo)
        self.ax[1].axis("off")
        self.canvas.draw()


root = tk.Tk()
app = EdgeDetApp(root)
root.mainloop()

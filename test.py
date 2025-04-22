import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("PDF files", "*.pdf")]
    )
    return file_path

# Exemple d'utilisation
file = select_file()
print("Fichier sélectionné :", file)

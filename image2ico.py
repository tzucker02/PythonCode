from pathlib import Path

from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox


def choose_files():
    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="Choose an image file",
        initialdir=r"c:\users\owner\pictures",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp *.ico"),
            ("All files", "*.*"),
        ],
    )

    if not input_path:
        root.destroy()
        return None, None

    default_output = str(Path(input_path).with_suffix(".ico"))
    output_path = filedialog.asksaveasfilename(
        title="Save ICO file as",
        initialdir=str(Path(input_path).parent),
        initialfile=Path(default_output).name,
        defaultextension=".ico",
        filetypes=[("ICO files", "*.ico")],
    )

    root.destroy()
    return input_path, output_path

def convert_image_to_ico():
    input_path, output_path = choose_files()
    if not input_path or not output_path:
        return

    img = Image.open(input_path).convert("RGBA")  # Ensure the image has an alpha channel
    # Saving with the specified size is common for ICO files
    img.save(output_path, format='ICO', sizes=[(256, 256)])
    messagebox.showinfo("Conversion complete", f"Saved icon to:\n{output_path}")

# Usage example:
convert_image_to_ico()
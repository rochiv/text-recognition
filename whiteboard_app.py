import tkinter as tk
from tkinter.colorchooser import askcolor
import datetime
from PIL import Image
import os


class WhiteboardApp:
    def __init__(self, root):
        self.prev_x = None
        self.prev_y = None
        self.line_width_slider = None
        self.root = root
        self.root.title("Whiteboard")
        self.root.geometry("800x600")

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.is_drawing = False
        self.drawing_color = "black"
        self.line_width = 4

        self.setup_controls()
        self.bind_events()

    def setup_controls(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side="top", fill="x")

        color_button = tk.Button(controls_frame, text="Change Color", command=self.change_pen_color)
        clear_button = tk.Button(controls_frame, text="Clear Canvas", command=lambda: self.canvas.delete("all"))
        save_button = tk.Button(controls_frame, text="Save Screenshot", command=self.save_screenshot)  # Save button

        color_button.pack(side="left", padx=5, pady=5)
        clear_button.pack(side="left", padx=5, pady=5)
        save_button.pack(side="left", padx=5, pady=5)  # Pack the save button

        line_width_label = tk.Label(controls_frame, text="Line Width:")
        line_width_label.pack(side="left", padx=5, pady=5)

        self.line_width_slider = tk.Scale(controls_frame, from_=1, to=10, orient="horizontal",
                                          command=self.change_line_width)
        self.line_width_slider.set(self.line_width)
        self.line_width_slider.pack(side="left", padx=5, pady=5)

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.is_drawing = True
        self.prev_x, self.prev_y = event.x, event.y

    def draw(self, event):
        if self.is_drawing:
            current_x, current_y = event.x, event.y
            self.canvas.create_line(self.prev_x, self.prev_y, current_x, current_y, fill=self.drawing_color,
                                    width=self.line_width, capstyle=tk.ROUND, smooth=True)
            self.prev_x, self.prev_y = current_x, current_y

    def stop_drawing(self, event):
        self.is_drawing = False

    def change_pen_color(self):
        color = askcolor()[1]
        if color:
            self.drawing_color = color

    def change_line_width(self, value):
        self.line_width = int(value*2)

    def save_screenshot(self):
        filename_ps = "screenshot_temp.ps"
        filename_png = datetime.datetime.now().strftime("screenshot_%Y-%m-%d_%H-%M-%S.png")

        # Save the current canvas content as a PostScript file
        self.canvas.postscript(file=filename_ps, colormode='gray')
        self.convert_ps_to_png(filename_ps, filename_png)
        os.remove(filename_ps)
        print(f"Screenshot saved as {filename_png}")

    from PIL import Image

    @staticmethod
    def convert_ps_to_png(input_ps_path, output_png_path, size=(28, 28)):
        with Image.open(input_ps_path) as img:
            img = img.convert('L')
            img_resized = img.resize(size)
            img_resized.save(output_png_path)

        return img_resized


if __name__ == "__main__":
    widget = tk.Tk()
    app = WhiteboardApp(widget)
    widget.mainloop()

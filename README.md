# Whiteboard App

The Whiteboard app is a simple, interactive drawing application built with Python's Tkinter library. It allows users to
draw freely with the mouse on a canvas, change the drawing color, adjust the line width, clear the canvas, and save
screenshots of their drawings.

## Features

- **Freehand Drawing:** Click and drag the mouse to start drawing.
- **Color Picker:** Change the color of your pen.
- **Line Width Adjustment:** Adjust the thickness of your drawing line.
- **Clear Canvas:** Erase all drawings from the canvas.
- **Save Screenshot:** Save your drawing as a 28x28 (default) grayscale PNG image.

## Installation

Before running the Whiteboard app, you need to ensure you have Python and PIL (Python Imaging Library) installed on your
system. If not, you can install Python from [python.org](https://www.python.org/) and PIL (Pillow) using pip.

### Install Pillow

Pillow is a fork of PIL and provides easy-to-use image processing capabilities. Install it using pip:

```sh
pip install Pillow
```

## Running the App

To run the Whiteboard app, navigate to the app's directory in your terminal or command prompt and execute the Python
script:

```sh
python whiteboard_app.py
```

## Usage

Upon launching the app, you'll be presented with a white canvas where you can start drawing immediately. Use the
controls at the top of the window to:

- **Change Color:** Opens a color chooser dialog to select a new pen color.
- **Clear Canvas:** Clears the current drawing from the canvas, giving you a blank slate.
- **Save Screenshot:** Saves the current canvas drawing as a grayscale PNG image in the current directory. The saved
  file will be named with the format `screenshot_YYYY-MM-DD_HH-MM-SS.png`.

## Requirements

- Python 3.x
- Tkinter (usually comes with Python)
- Pillow

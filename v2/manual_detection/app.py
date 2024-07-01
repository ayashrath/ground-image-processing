"""
The Flask Application
"""

import os
import numpy as np
from create_img import drawSquareGrid
from flask import Flask, render_template, send_from_directory, request, jsonify


# constants
BACKUP_PATH = "../backups"
UNIT = "Celcius"


app = Flask(__name__)


# index
@app.route("/")
def index():
    # checks if dir -> just in case
    dirs = [d for d in os.listdir(BACKUP_PATH) if os.path.isdir(os.path.join(BACKUP_PATH, d))]
    dirs.sort(reverse=True)
    return render_template("index.html", directories=dirs)


# view image page
@app.route("/view/<directory>")
def view_image(directory):
    image_path = os.path.join(BACKUP_PATH, directory, "img.jpg")
    if os.path.exists(image_path):
        return render_template("view_image.html", directory=directory)
    else:
        return "Image not found", 404  # incase, though it should not happen


# send img file
@app.route("/images/<directory>/<filename>")
def serve_image(directory, filename):
    return send_from_directory(os.path.join(BACKUP_PATH, directory), filename)


# get temp stuff
@app.route("/process_coordinates", methods=["POST"])
def process_coordinates():
    data = request.json
    coordinates = data.get("coordinates")
    directory = data.get("directory")

    # strip json - not stripped before, for debugging purposes
    # this is a list of the rows
    cell_array = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    raw_temp_array = np.loadtxt(os.path.join(BACKUP_PATH, directory, "raw_temp_array.txt"))

    for cell in coordinates:
        # the coordinates row and column keys are not 0 start while the x and y are
        # in these arrays the first index that is used is for the row while the second is for the column
        cell_array[cell["row"] - 1][cell["col"] - 1] = raw_temp_array[cell["y"]][cell["x"]]

    if UNIT == "Celcius":
        drawSquareGrid(os.path.join(BACKUP_PATH, directory), cell_array, "C")
    elif UNIT == "Kelvin":
        drawSquareGrid(os.path.join(BACKUP_PATH, directory), cell_array, "K")
    else:
        raise ValueError

    return jsonify({"status": "success", "message": "Coordinates received"}), 200


# debug statement
if __name__ == "__main__":
    app.run(debug=True)

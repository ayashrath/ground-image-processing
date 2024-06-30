"""
The Flask Application
"""

import os
import sys
from flask import Flask, render_template, send_from_directory, request, jsonify

sys.path.append("../")
import input_handler as ih

app = Flask(__name__)

BACKUP_PATH = os.path.join("../", ih.BACKUP_PATH)

# index
@app.route('/')
def index():
    # checks if dir -> just in case
    dirs = [d for d in os.listdir(BACKUP_PATH) if os.path.isdir(os.path.join(BACKUP_PATH, d))]
    dirs.sort(reverse=True)
    return render_template('index.html', directories=dirs)


# view image
@app.route('/view/<directory>')
def view_image(directory):
    image_path = os.path.join(BACKUP_PATH, directory, 'img.jpg')
    if os.path.exists(image_path):
        return render_template('view_image.html', directory=directory)
    else:
        return "Image not found", 404  # incase, though it should not happen


if __name__ == '__main__':
    app.run(debug=True)

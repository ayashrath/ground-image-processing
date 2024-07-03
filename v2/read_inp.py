"""
Reads in the inputs and then stores them on the drive
"""

import os
import sys
import time
import socket
import numpy as np
import cv2
from detect import detect_cell_temps

sys.path.append(os.path.join(os.path.dirname(__file__), "manual_detection"))
from create_img import drawSquareGrid


# Constants
BACKUP_PATH = "./backups"
UNIT = "Celcius"
SOURCE_IP_ADDR = ""
UDP_PORT = 53
ARRAY_WIDTH = 80  # as img is 80x64
ARRAY_HEIGHT = 64
ARRAY_ELEM_BYTE_SIZE = 4  # as float


# establish conn
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SOURCE_IP_ADDR, UDP_PORT))


while True:
    # get data and make it usable
    data, addr = sock.recvfrom(ARRAY_WIDTH * ARRAY_HEIGHT * ARRAY_ELEM_BYTE_SIZE)
    thermal_array = np.frombuffer(data, dtype=np.int).reshape((ARRAY_HEIGHT, ARRAY_WIDTH))

    # store
    if UNIT == "Celcius":   # as the data from the sensor are in Kelvin
        thermal_array -= 273
    elif UNIT != "Kelvin":
        raise ValueError

    # make dir and cd into it
    unix_time = int(time.time())  # to only keep till the seconds
    new_path = os.path.join(BACKUP_PATH, str(unix_time))

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    else:  # it should not happen but whatever
        raise FileExistsError

    os.chdir(new_path)

    # save raw array
    np.savetxt("raw_temp_array.txt", thermal_array)

    # normalise and create image in Jet colourspace
    normalised_array = cv2.normalize(thermal_array, None, 0, 255, cv2.NORM_MINMAX)
    normalised_array = np.uint8(normalised_array)
    colourmapped_img = cv2.applyColorMap(normalised_array, cv2.COLORMAP_JET)

    img_path = os.path.join(os.getcwd(), "img.jpg")
    cv2.imwrite(img_path, colourmapped_img)

    # add auto code here when you make
    centres = detect_cell_temps(thermal_array)
    temp_2d = [[0 for i in range(5)] for j in range(5)]

    for key, val in data.items():
        ind = int(key) - 1
        temp_2d[ind//5][ind % 5] = thermal_array[val[1]][val[0]]  # as y axis represents rows

    drawSquareGrid("./auto_result.png", temp_2d, UNIT)


# manage exceptions and stuff

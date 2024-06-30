"""
Input classes
"""

import os
import time
from typing import Tuple
import socket
import numpy as np
import cv2


# Constants
BACKUP_PATH = "./backup"
UNIT = "Celcius"


class UDPConn:
    """
    The class that manages the UDP connection

    Parameters:
        - ip_addr (str): The IP address of the source
        - port (int): The port used for the UDP signal
        - array_size (tuple): The first element represents width, while the second represents the height
        - byte_size (int): The byte size of each of the element of the array that is being received
    """
    def __init__(self, ip_addr: str, port: int, array_size: Tuple[int, int] = (80, 64), byte_size: int = 4):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet, UDP
        self.sock.bind((ip_addr, port))

        self.width = array_size[0]
        self.height = array_size[1]
        self.byte_size = byte_size

    def get_buffer(self) -> np.ndarray:
        """
        Get stuff from stream

        Returns:
          - thermal_array (np.ndarray): The thermal array received
        """
        data, addr = self.sock.recvfrom(self.width * self.height * self.byte_size)
        thermal_array = np.frombuffer(data, dtype=np.int).reshape((self.height, self.width))

        return thermal_array


class Thermal_Store:
    """
    It helps store the thermal data, thus facilitates backup
    The data is stored in a folder with name from the current unix time
    It will contain 2 info files - the raw data and image file representing the data (JET)

    **Note:** The image file is created by normalising the data from the raw data array, so it can't be used to get the
    exact data info.

    Parameters:
      - temp_array (np.ndarray): The np array that contains the pixel temp data from the thermal camera input
      - convert_to_C (bool) (default=True): The input is in Kelvin, so it converts to celcius if needed

    Exceptions:
      - FileExistsError if the folder for backup already exists (should not as it is done by using UNIX time)
    """
    def __init__(self, temp_array: np.ndarray, convert_to_C: bool = True):
        if UNIT == "Celcius":
            temp_array = temp_array - 273  # from K to C
        elif UNIT != "Kelvin":
            raise ValueError

        # Raw array
        self.raw_temp_array = temp_array

        # Normalised, Colourmaped Array
        normalised_array = cv2.normalize(self.raw_data_store, None, 0, 255, cv2.NORM_MINMAX)
        normalised_array = np.uint8(normalised_array)
        self.colourmapped_temp_array = cv2.applyColorMap(normalised_array, cv2.COLORMAP_JET)

        # Make and cd into Dir
        unix_time = int(time.time())  # only keeps till the seconds
        new_path = os.path.join(BACKUP_PATH, str(unix_time))

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        else:  # it should not happen but whatever
            raise FileExistsError

        os.chdir(new_path)

    def backup(self) -> None:
        """
        Makes the needed backups for the raw data and the image file
        """

        # raw data
        np.savetxt("raw_temp_array.txt", self.raw_temp_array)

        # image
        img_path = os.path.join(os.getcwd(), "infra.jpg")
        cv2.imwrite(img_path, self.colourmapped_temp_array)

        # !!! Have the Auto Code Here if you make it - SQL DB STORE FOR IT!!!


class Thermal_Retrieve:
    """
    Its main job is to retrieve data that is stored in the disk

    Parameters:
      - dir_name: the unix time name which represents an image dir

    Exceptions:
      - FileNotFoundError if the Unix time dir or any of its files are not found
    """

    def __init__(self, dir_name: str):
        full_path = os.path.join(BACKUP_PATH, dir_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError

        os.chdir(full_path)

        # The raw data
        self.raw_data_array = np.loadtxt("raw_temp_array.txt")  # can raise FileNotFound

        # The image
        self.img = cv2.imread("infra.jpg")
        if self.img is None:
            raise FileNotFoundError

    def get_raw_temp_data(self) -> np.ndarray:
        """
        Returns:
          - raw_data_array: The pixel temp values of the thermal image
        """
        return self.raw_data_array

    def get_img(self) -> np.ndarray:
        """
        Returns:
          - img: The thermal img array
        """
        return self.img

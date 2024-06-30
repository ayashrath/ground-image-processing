"""
Input classes
"""

import os
import time
from typing import Tuple
import socket
import numpy as np
import cv2


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


class Thermal_Input:
    """
    It helps store the thermal data, thus facilitates backup "and restore of data"
    The data is stored in a folder with name from the current unix time
    It will contain 3 info files - the raw data, the unit of the data and finally a image file representing the data

    **Note:** The image file is created by normalising the data from the raw data array, so it can't be used to get the
    exact data info

    Parameters:
      - temp_array (np.ndarray): The np array that contains the pixel temp data from the thermal camera input
      - convert_to_C (bool) (default=True): The input is in Kelvin, so it converts to celcius if needed
      - path (str) (default="./backups"): The path where the backup for data of object of this class will be stored
      - colour_map (int) (default=cv2.COLORMAP_JET)

    Exceptions:
      - FileExistsError if the folder for backup already exists (should not as it is done by using UNIX time)
    """
    def __init__(
        self, temp_array: np.ndarray, convert_to_C: bool = True, path: str = "./backups",
        colour_map: int = cv2.COLORMAP_JET
    ):

        # Unit
        self.UNIT = "Kelvin"
        if convert_to_C:
            temp_array = temp_array - 273
            self.UNIT = "Celcius"

        # Raw array
        self.raw_temp_array = temp_array

        # Normalised, Colourmaped Array
        normalised_array = cv2.normalize(self.raw_data_store, None, 0, 255, cv2.NORM_MINMAX)
        normalised_array = np.uint8(normalised_array)
        self.colourmapped_temp_array = cv2.applyColorMap(normalised_array, colour_map)

        # Make and cd into Dir
        unix_time = int(time.time())  # only keeps till the seconds
        new_path = os.path.join(path, str(unix_time))

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        else:  # it should not happen but whatever
            raise FileExistsError

        os.chdir(new_path)

    def backup(self) -> None:
        """
        Makes the needed backups for the raw data, unit and the image file
        """

        # raw data
        np.savetxt('raw_temp_array.txt', self.raw_temp_array)

        # unit
        with open("unit.txt", "w") as file:
            file.write(self.UNIT)

        # image
        img_path = os.path.join(os.getcwd(), "infra.jpg")
        cv2.imwrite(img_path, self.colourmapped_temp_array)

        # !!! Have the Auto Code Here if you make it !!!

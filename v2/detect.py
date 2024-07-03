"""
Autodetect the grids and and provides the detected temps from the thermal image

The idea is that the scene will only have 1 main heat source, while the rest
of the background will be mostly room temps (so blue).

Thus by using basic contour detection the main square (+heat bleed) can be seen
The thing is the heat bleed needs to be eliminatated, so the idea is
to take the centre of the detected contour and assume it to be the same as 
the 3rd row and colomn cell of the square grid, i.e, assuming the head bleed is 
evenly done. And then using expected pixel length of the square grid, everything is mapped out
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


# constants
MIN_CONTOUR_AREA = 200


def detect_cell_temps(raw_temp_array: np.ndarray) -> np.ndarray:
    """
    It detects the temp of each cell using the raw_temp matrix that the sensor returns

    Parameter:
      - raw_temp_array (np.ndarray): The array that the sensor provides

    Returns:
      - temp_array (np.ndarray): Array containing temps of the individual cells 
    """

    # is in greyscale
    img = cv2.normalize(raw_temp_array, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(thresh, 30, 100)

    """plt.imshow(blur, cmap="jet")
    plt.title('Blurred Image')
    plt.show()"""

    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = [c for c in cnts if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    cnts_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(cnts_image, cnts, -1, (0, 255, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cnts_image)
    plt.title('Contours')
    plt.show()

    return np.array(cnts_image)


# testing code
if __name__ == "__main__":
    # Sample thermal data - generated using DALL-E, and is not exact resolution of the Heimann camera output
    inp = cv2.imread("detect-test.png", cv2.IMREAD_GRAYSCALE)
    detect_cell_temps(inp)

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
import random


# constants
MIN_CONTOUR_AREA = 200
RATIO_BOUNDS = (0.85, 1.15)  # if a square then the ideal would be 1


def detect_cell_temps(raw_temp_array: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    It detects the temp of each cell using the raw_temp matrix that the sensor returns.

    Parameter:
      - raw_temp_array (np.ndarray): The array that the sensor provides
      - debug (boolen)(=False): If true it show the center points on the image

    Returns:
      - centres (dict): Dict where the keys are the cell number and the value is tuple for coordinate for cell's centre

    Raises:
      - Exception due to not detecting the needed contour
    """

    # normalise array to represent an image
    img = cv2.normalize(raw_temp_array, None, 0, 255, cv2.NORM_MINMAX)  # img is in greyscale
    img = np.uint8(img)

    # contour detection
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 30, 100)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # reducing it to only 1 contour
    if len(cnts) > 1:
        min_area = img.size + 1
        min_cnt = None
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if RATIO_BOUNDS[1] < aspect_ratio < RATIO_BOUNDS[-1]:
                continue

            if min_area > cv2.contourArea(cnt) and cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                min_cnt = cnt
        cnts = [min_cnt]

    if cnts[0] is None:
        raise Exception("Couldn't detect any contours")

    # cell_no: (x, y)
    centres = {
        "1": (0, 0),
        "2": (0, 0),
        "3": (0, 0),
        "4": (0, 0),
        "5": (0, 0),
        "6": (0, 0),
        "7": (0, 0),
        "8": (0, 0),
        "9": (0, 0),
        "10": (0, 0),
        "11": (0, 0),
        "12": (0, 0),
        "13": (0, 0),
        "14": (0, 0),
        "15": (0, 0),
        "16": (0, 0),
        "17": (0, 0),
        "18": (0, 0),
        "19": (0, 0),
        "20": (0, 0),
        "21": (0, 0),
        "22": (0, 0),
        "23": (0, 0),
        "24": (0, 0),
        "25": (0, 0),
    }

    # am using the median coordinates of the edges to account for the imperfections that exist in the edges and
    # the corners. I could try to make a perfect square instead, but that could be finicy and this seems 
    # to also be a simpler solution to the problem
    flatten_cnts = [(point[0][0], point[0][1]) for point in cnts[0]]

    x_array = sorted([x for x, y in flatten_cnts])
    y_array = sorted([y for x, y in flatten_cnts])

    left_edge_x = x_array[:len(x_array)//2]
    right_edge_x = x_array[len(x_array)//2:]
    top_edge_y = y_array[:len(y_array)//2]
    bottom_edge_y = y_array[len(y_array)//2:]

    c_13_x = int((np.median(left_edge_x) + np.median(right_edge_x))/2)
    c_13_y = int((np.median(top_edge_y) + np.median(bottom_edge_y))/2)

    centres["13"] = c_13_x, c_13_y

    # the reference stuff would be the x coord of left edge and y cood of top edge
    x, y = int(np.median(left_edge_x)), int(np.median(top_edge_y))
    x_13, y_13 = centres["13"]

    # for 11, 12, 14 and 15
    x_diff = x_13 - x
    centres["11"], centres["12"], centres["14"], centres["15"] = (
        (x_diff // 5 + x, y_13),
        (x_diff * 3 // 5 + x, y_13),
        (x_diff * 7 // 5 + x, y_13),
        (x_diff * 9 // 5 + x, y_13),
    )

    # for 3, 8, 18, 23
    y_diff = y_13 - y
    if y < y_13:
        centres["3"], centres["8"], centres["18"], centres["23"] = (
            (x_13, y_diff // 5 + y),
            (x_13, y_diff * 3 // 5 + y),
            (x_13, y_diff * 7 // 5 + y),
            (x_13, y_diff * 9 // 5 + y),
        )

    # for rest
    for ref_cell in [11, 12, 14, 15]:
        col_cell = ref_cell + np.array([-10, -5, 5, 10])
        col_cell = [str(i) for i in col_cell]
        ref_x, ref_y = centres[str(ref_cell)]
        y_diff = ref_y - y
        centres[col_cell[0]], centres[col_cell[1]], centres[col_cell[2]], centres[col_cell[3]] = (
            (ref_x, y_diff // 5 + y),
            (ref_x, y_diff * 3 // 5 + y),
            (ref_x, y_diff * 7 // 5 + y),
            (ref_x, y_diff * 9 // 5 + y),
        )

    # display
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)

    temp = [
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (0, 0, 255),
        ]

    for cell_no in centres:
        cv2.circle(img, centres[cell_no], 1, random.choice(temp), -1)

    if debug:
        print("Centres:", centres)
        plt.imshow(img)
        plt.title('Contours')
        plt.show()

    return centres


# testing code
if __name__ == "__main__":
    # Sample thermal data - generated using DALL-E, and is not exact resolution of the Heimann camera output
    inp = cv2.imread("detect-test.png", cv2.IMREAD_GRAYSCALE)
    detect_cell_temps(inp, debug=True)

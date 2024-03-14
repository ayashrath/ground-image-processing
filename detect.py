"""
Ground Image Processing Algorithms
"""

import subprocess
from typing import Tuple, List
import numpy as np
import cv2
from imutils import contours


def show_img(img: np.ndarray, title: str = "Image") -> None:
    """
    Displays image using OpenCV

    Parameter:
        - img: The image
        - title: The window title
    """

    cv2.imshow(title, img)
    cv2.waitKey(5000)


def gen_video(path: str, frames: List[np.ndarray], fps: int = 30, frame_size: Tuple[int, int] = (1920, 1080), bin: str = "/bin/ffmpeg", end_extra: bool = False, image_scale: str = "grey") -> None:
    """Generate Video with list of Image, using ffmpeg

    Parameter:
        path: Path where video will be saved
        frames: List of images to make the video
        fps: The fps of the video
        frame_size: The resolution of the video
        bin: The path where the ffmpeg binary
        end_extra: If you want to have the last frame of the video to last a bit longer
    """

    bin = "C:\\Users\\ayash\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-6.1.1-full_build\\bin\\ffmpeg.exe"

    if image_scale == "rgb":
        pix_fmt = "rgb24"
    elif image_scale == "grey":
        pix_fmt = "gray"
    else:
        raise ValueError("Unsupported image format")

    # ffmpeg command to create a video from frames
    ffmpeg_cmd = [
        bin,
        "-y",  # Overwrite output files without asking
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{frame_size[0]}x{frame_size[1]}",
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        "-i", "-",
        "-an",  # No audio
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        path
    ]

    # Open a subprocess to execute the ffmpeg command and also silence output and error
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write frames to the subprocess
    for frame in frames:
        ffmpeg_process.stdin.write(cv2.resize(frame, frame_size).tobytes())

    # Extend last frames if needed
    if end_extra:
        for i in range(5):
            ffmpeg_process.stdin.write(cv2.resize(frames[-1], frame_size).tobytes())

    # Close the subprocess
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


class Image:
    """Represents the image (convert into greyscale is not greyscale)

    Parameter:
        img: The image

    Error:
        - ValueError if the image is neither RGB/BGR/greyscale
    """

    def __init__(self, img: np.ndarray) -> None:
        if len(img.shape) == 3 and img.shape[2] == 3:  # If it is an BGR image
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:  # Already is a greyscale image
            self.image = img
        else:
            raise ValueError("Unsupported image format")

    def get_img(self) -> np.ndarray:
        """
        Access image

        Return:
            - self.image: The image
        """

        return self.image

    def resize(self, x_len: int, y_len: int) -> np.ndarray:
        """
        Resize image

        Return:
            - resized image
        """

        return cv2.resize(self.get_img(), (x_len, y_len))

    def threshold(self) -> np.ndarray:
        """
        Threshold of image

        Return:
            - thresh: The threshold image
        """

        # blur to smoothen out the edges which improve the adaptive threshold results
        blur = cv2.GaussianBlur(self.get_img(), (7, 7), 0)

        # Explain the threshold algorithms
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 7)
        _, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_OTSU)

        return thresh

    def condition(self, cnts: np.ndarray, min: int = 100000, max: int = 200000, max_count: int = 35, min_count: int = 25) -> Tuple[int, int]:
        """
        Adjust variables to ensure the squares get tracked

        Parameters:
            - cnts: array of counters
            - min: The min value from which the calculations start from
            - max: The max value from which the calculation end at
            - max_count: The max count of squares (contours) allowed here
            - min_count: The min count of squares (contours) allowed here

        Return:
            - (max_value, min_value): The values that satisfy the situation
        """

        min_val = 0
        max_val = max-min
        diff = 1000

        # if the input has no contours
        if len(cnts) < min_count:
            return 0, 0

        # Explain Loop
        while min_val <= min:
            count = 0
            for c in cnts:
                area = cv2.contourArea(c)
                if max_val > area > min_val:
                    count += 1

            if max_count >= count >= min_count:
                return max_val, min_val

            min_val += diff
            max_val += diff

        # If the needed <>_count conditions couldn't be satisfied
        return 0, 0

    def filter_boxes(self, thresh: np.ndarray) -> np.ndarray:  # change
        """
        Morph the image

        Parameter:
            - thresh: The threshold image

        Return:
            - morph: The morphed image
        """

        # Fix spot issues that are introduced due to thresholding
        morph = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=0)
        morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=0)

        # Fix issues that may be present on both the axes due to having non-straight lines in the grid
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, vertical_kernel, iterations=0)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, horizontal_kernel, iterations=0)

        return morph

    def centre(self, c) -> Tuple[int, int]:  # change it such that it can find all the contours itself, and not one at a time
        """
        Get the centre of contours

        Parameters:
            - c: a single contour

        Return:
            - (cX, cY) - x and y coordinates of centre of contour
        """

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return cX, cY

    def get_squares(self, morph) -> List[np.ndarray]:
        """
        Get the squares from the morphed image

        Parameters:
            - morph: Morphed image

        Return:
            - list of contours
        """
        # Sort by top to bottom and each row by left to right
        invert = 255 - morph
        cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 2:
            cnts = cnts[0]
        else:
            cnts = cnts[1]

        try:
            (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
        except ValueError:
            cnts = []

        cells = []
        row = []

        max_val, min_val = self.condition(cnts)

        for (i, c) in enumerate(cnts, 1):
            area = cv2.contourArea(c)
            if max_val > area > min_val:
                row.append(c)
                if i % 5 == 0:
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    for c in cnts:
                        cells.append(c)
                    row = []

        count = len(cells)

        if __name__ != "__main__":
            print("Cell Count", count)

        return cells

    def detect_squares(self) -> List[np.ndarray]:
        """
        The entire algorithm

        Return:
            - the_squares: The detected contors
        """

        thresh = self.threshold()
        morph = self.filter_boxes(thresh)
        the_squares = self.get_squares(morph)

        return the_squares


# Test
if __name__ == '__main__':
    for name in ["normal", "left", "right", "down", "up"]:
        print("Current:", name)
        img = Image(cv2.imread("img/" + name + ".jpg"))
        squares = img.detect_squares()

        for cell in squares:
            cX, cY = img.centre(cell)
            cv2.drawContours(img.get_img(), [cell], -1, (0, 255, 0), -1)
            cv2.circle(img.get_img(), (cX, cY), 7, (0, 0, 255), -1)

        show_img(img.resize(800, 800), title='frame')

    cv2.destroyAllWindows()

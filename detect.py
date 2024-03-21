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


def gen_video(
    path: str, frames: List[np.ndarray], fps: int = 30, frame_size: Tuple[int, int] = (1920, 1080),
    bin: str = "/bin/ffmpeg", end_extra: bool = False, image_scale: str = "grey"
        ) -> None:
    """Generate Video with list of Image, using ffmpeg

    Parameter:
        path: Path where video will be saved
        frames: List of images to make the video
        fps: The fps of the video
        frame_size: The resolution of the video
        bin: The path where the ffmpeg binary
        end_extra: If you want to have the last frame of the video to last a bit longer
    """

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

        # The gaussian adaptive threshold deals with the shadows while otsu helps to get the grid out of that
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 7)
        _, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_OTSU)

        return thresh

    def condition(
        self, cnts: np.ndarray, threshold_percent=10, max_count: int = 27, min_count: int = 25
            ) -> Tuple[int, int]:
        """
        Adjust variables to ensure the squares get tracked

        Parameters:
            - cnts: array of counters
            - threshold_percent: The percentage difference allowed between countor areas
            - max_count: The max count of squares (contours) allowed here
            - min_count: The min count of squares (contours) allowed here

        Return:
            - cnts: The required square cnts
        """

        areas = []

        for c in cnts:
            area = cv2.contourArea(c)
            areas.append(area)

        areas.sort()
        groups = []
        current_group = [areas[0]]

        for i in range(1, len(areas)):
            percent_diff = abs(areas[i] - areas[i - 1]) / areas[i - 1] * 100
            if percent_diff <= threshold_percent:
                current_group.append(areas[i])
            else:
                groups.append(current_group)
                current_group = [areas[i]]

        groups.append(current_group)
        print(groups)
        final = []

        for i in groups:
            print(len(i))
            if max_count >= len(i) >= min_count:
                final.append(i)

        print(final)
        return final[-1]

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

    def centre(self, c) -> Tuple[int, int]:
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

        cnts = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 2:
            cnts = cnts[0]
        else:
            cnts = cnts[1]

        try:
            (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
        except ValueError:
            cnts = []

        cells = self.condition(cnts)

        if __name__ == "__main__":
            count = len(cells)
            print(count, cells)

        return cells

    def detect_squares(self) -> List[np.ndarray]:
        """
        The entire algorithm

        Return:
            - the_squares: The detected contors
        """

        thresh = self.threshold()
        # show_img(cv2.resize(thresh, (800, 800)))
        morph = self.filter_boxes(thresh)
        # temp = cv2.resize(morph, (800, 800))
        # show_img(temp)
        the_squares = self.get_squares(morph)

        return the_squares


# Test
if __name__ == '__main__':
    for name in ["normal", "left", "right", "down", "up"]:
        print("Current:", name)
        img = Image(cv2.imread("img/" + name + ".jpg"))
        squares = img.detect_squares()

        img_bgr = cv2.imread("img/" + name + ".jpg")

        for cell in squares:
            cX, cY = img.centre(cell)
            cv2.drawContours(img_bgr, [cell], -1, (0, 255, 0), -1)
            cv2.circle(img_bgr, (cX, cY), 7, (0, 0, 255), -1)

        img_bgr = cv2.resize(img_bgr, (800, 800))

        show_img(img_bgr, title='frame')
        quit()

    cv2.destroyAllWindows()

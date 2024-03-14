"""
Ground Image Processing Algorithms
"""

import subprocess
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


def gen_video(path: str, frames: list(np.ndarray), fps: int = 30, frame_size: (int, int) = (1920, 1080), bin: str = "/bin/ffmpeg", end_extra: bool = False) -> None:
    """Generate Video with list of Image, using ffmpeg

    Parameter:
        path: Path where video will be saved
        frames: List of images
        fps: The fps of the video
        frame_size: The resolution of the video
        bin: The path where the ffmpeg binary can be found
        end_extra: If you want to have the last frame of the video to last a bit longer
    """

    bin = "C:\\Users\\ayash\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-6.1.1-full_build\\bin\\ffmpeg.exe"

    # FFmpeg command to create a video from frames
    ffmpeg_cmd = [
        bin,
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frame_size[0]}x{frame_size[1]}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-an',  # No audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        path
    ]

    # Open a subprocess to execute the FFmpeg command
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write frames to the subprocess
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(cv2.resize(frame, frame_size).tobytes())

    if end_extra:
        for i in range(5):
            ffmpeg_process.stdin.write(cv2.resize(frames[-1], frame_size).tobytes())

    # Close the subprocess
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


class Image:
    """Represents the image

    Parameter:
        img: The image
    """
    def __init__(self, img: np.ndarray) -> None:
        self.image: np.ndarray = img
        # Maybe convert to greyscale if it is not already

    def get_img(self) -> np.ndarray:
        """ Access image

        Return:
            - self.image: The image
        """

        return self.image

    def threshold(self) -> np.ndarray:
        """ Threshold of image

        Return:
            - thresh: The threshold image
        """

        blur = cv2.GaussianBlur(self.get_img(), (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 7)
        _, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_OTSU)

        return thresh

    def condition(self, cnts, min: int = 100000, max: int = 200000, max_count: int = 35, min_count: int = 25) -> (int, int):
        """ Adjust variables to ensure the squares get tracked

        Parameters:
            - cnts:
            - min:
            - max:
            - max_count:
            - min_count:

        Return:
            - ...
        """

        min_val = 0
        max_val = max-min
        diff = 1000

        if len(cnts) < min_count:
            return 0, 0

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

        return 0, 0

    def filter_boxes(self, thresh: np.ndarray) -> np.ndarray:
        """
        Morph the image

        Parameter:
            - thresh: The threshold image

        Return:
            - morph: The morphed image
        """

        # Filter out all numbers and noise to isolate only boxes
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        morph = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        if len(cnts) == 2:
            cnts = cnts[0]
        else:
            cnts = cnts[1]

        max_val, min_val = self.condition(cnts)

        for c in cnts:
            area = cv2.contourArea(c)
            if max_val > area > min_val:  # CHANGE THIS LATER FOR VIDEO THING
                cv2.drawContours(morph, [c], -1, (0, 255, 0), 3)

        morph = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)

        # CHANGE START
        morph = cv2.dilate(morph, np.ones((5, 5), np.uint8), iterations=0)
        morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=0)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, vertical_kernel, iterations=0)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, horizontal_kernel, iterations=0)
        # CHANGE STOP

        return morph

    def centre(self, c) -> (int, int):  # change it such that it can find all the contours itself, and not one at a time
        """
        Get the centre of contours

        Parameters:
            - c: a single contour
        """

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return cX, cY

    def get_squares(self, morph):
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

        sudoku_rows = []
        row = []

        max_val, min_val = self.condition(cnts)

        for (i, c) in enumerate(cnts, 1):
            area = cv2.contourArea(c)
            if max_val > area > min_val:
                row.append(c)
                if i % 9 == 0:
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    sudoku_rows.append(cnts)
                    row = []

        count = 0
        for row in sudoku_rows:
            count += len(row)

        if __name__ == "__main__":
            print("CNTS Count", count)

        return sudoku_rows

    def detect_squares(self):
        thresh = self.threshold()
        temp = cv2.resize(thresh, (800, 800))
        show_img(temp, title="thresh")
        morph = self.filter_boxes(thresh)
        temp = cv2.resize(morph, (800, 800))
        show_img(temp, title="morph")
        the_squares = self.get_squares(morph)
        # results = self.sort_squares_view(the_squares)
        return the_squares


# Test
if __name__ == '__main__':
    for name in ["normal", "left", "right", "down", "up"]:
        print("Current:", name)
        img = Image("img/" + name + ".jpg")
        squares = img.detect_squares()
        quit()
    cv2.destroyAllWindows()

"""
Ground Image Processing Algorithms
"""

import subprocess
import numpy as np
import cv2
from imutils import contours


class ImageError(Exception):
    pass


class Image:
    """What does it represent"""
    def __init__(self, path):
        if __name__ == "__main__":
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = path

        if isinstance(self.image, bool):
            print("Unable to read image")
            raise ImageError("Can't read the Image")
            quit()

    def show(self, img, title="Image"):
        """use_plt: if true uses matplotlib image show, else OpenCV's"""
        if __name__ != "__main__":
            cv2.imshow(title, img)
            # cv2.waitKey(5000)

    def video(self, path, frames, end_extra=False):
        """"""
        fps = 30
        frame_size = (1920, 1080)

        bin = "C:\\Users\\ayash\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-6.1.1-full_build\\bin\\ffmpeg.exe"

        # FFmpeg command to create a video from frames
        ffmpeg_cmd = [
            bin,
            '-y',  # Overwrite output files without asking
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{frame_size[0]}x{frame_size[1]}',  # Width x Height
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-an',  # No audio
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            path
        ]

        # Open a subprocess to execute the FFmpeg command, shell = true for windows
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
            for i in range(3):
                ffmpeg_process.stdin.write(cv2.resize(frames[-1], frame_size).tobytes())

        # Close the subprocess
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

        # Print a message indicating that the video has been saved
        print("Video saved.")

    def threshold(self):
        blur = cv2.GaussianBlur(self.image, (7, 7), 0)
        # self.show(blur, title="blur")

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 7)
        # self.show(title="thresh")
        _, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_OTSU)
        # self.show(thresh, title="otsu")

        return thresh

    def condition(self, cnts, min=100000, max=200000, max_count=35, min_count=25):  # Change + Optimise
        if __name__ == "__main__":
            # print(area)
            pass

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

        # return False
        return 0, 0

    def filter_boxes(self, thresh):
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

        # self.show(morph, title="Contours on Otsu")
        morph = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
        # quit()

        # Fix horizontal and vertical lines

        # morph = cv2.dilate(morph, np.ones((5, 5), np.uint8), iterations=1)  # CHANGE
        # morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=1)  # CHANGE

        # IGNORE
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # CHANGE
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, vertical_kernel, iterations=0)   # CHANGE
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # CHANGE
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, horizontal_kernel, iterations=0)  # CHANGE
        # STOP IGNORE

        # self.show(morph, title="morph")

        return morph

    def centre(self, c):
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

    def sort_squares_view(self, sudoku_rows):
        results = []

        # Iterate through each box
        for row in sudoku_rows:
            for c in row:
                mask = np.zeros(self.image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
                result = cv2.bitwise_and(self.image, mask)
                result[mask == 0] = 255
                # cv2.imshow('result', cv2.resize(result, (800, 800)))
                # cv2.waitKey(350)
                # results.append(result)

        # cv2.drawContours(self.image, contours, -1, (0, 255, 0), 3)

        # print("Number of Contours found = " + str(len(contours)))
        # self.show(self.image, title="Countours")
        # self.image = cv2.erode(self.image, None, iterations=1)

        return results

    def detect_squares(self):
        thresh = self.threshold()
        temp = cv2.resize(thresh, (800, 800))
        self.show(temp, title="thresh")
        morph = self.filter_boxes(thresh)
        temp = cv2.resize(morph, (800, 800))
        self.show(temp, title="morph")
        the_squares = self.get_squares(morph)
        # results = self.sort_squares_view(the_squares)
        return the_squares


# Test
if __name__ == '__main__':
    for name in ["normal", "left", "right", "down", "up"]:
        print("Current:", name)
        img = Image("img/" + name + ".jpg")
        squares = img.detect_squares()

        # img.video("vid/" + name + ".avi", squares)
        quit()
    cv2.destroyAllWindows()

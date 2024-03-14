"""
Aim: It makes use of <detect.py>  on live camera feed to test the program actively and save the result using ffmpeg
"""

import cv2
from detect import Image

vid = cv2.VideoCapture(0)  # Get input
frames = []  # Collect frames to save then as a file

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image(img)
    squares = img.detect_squares()

    for row in squares:
        for c in row:
            cX, cY = img.centre(c)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
            cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    frames.append(frame)

    # Press Q key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

img.video("vid/Detect.avi", frames)

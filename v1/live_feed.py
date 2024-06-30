"""
Aim: It makes use of <detect.py>  on live camera feed to test the program actively and save the result using ffmpeg
"""

import cv2
from detect import Image, gen_video

vid = cv2.VideoCapture(0)  # Get input
frames = []  # Collect frames to save then as a file

while (True):
    # Capture the video frame by frame
    ret, frame = vid.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image(img)
    cells = img.detect_squares(output=True, no_delay=True)  # change output= if you need to see diagnotic data

    for cell in cells:
        cX, cY = img.centre(cell)
        cv2.drawContours(frame, [cell], -1, (0, 255, 0), -1)
        cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)

    cv2.imshow('Video Frame', frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

    # Press Q key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

ffmpeg_loc = "C:\\Users\\ayash\\AppData\\Local\\Microsoft\\WinGet\\Packages\\\
Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-6.1.1-full_build\\bin\\ffmpeg.exe"

# gen_video("vid/Detect.avi", frames, bin=ffmpeg_loc, output_suppress=False)

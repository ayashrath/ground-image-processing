from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# constant
SIZE = 5
DIMENSIONS = 800
FONT_SIZE = 15
FONT_TFF = "./JetBrainsMonoRegular.ttf"


def drawSquareGrid(path: str, array: np.ndarray, unit: str) -> None:
    """
    Draws a square grid with each cell containing a value, thus providing a way to visually store data for the project

    Parameters:
      - path (str): The place where the image would be saved
      - array (np.ndarray): The array that contains the values for each cell - in form as shown in notes

    Note:
    - the input array example. The array is a array of rows
    - cell_array = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    """
    img = Image.new(mode="L", size=(DIMENSIONS, DIMENSIONS), color=255)

    draw = ImageDraw.Draw(img)

    # verticals
    for x in range(0, DIMENSIONS, DIMENSIONS//SIZE):
        y_st = 0
        y_end = DIMENSIONS

        draw.line(((x, y_st), (x, y_end)), fill=0)

    # horizontals
    for y in range(0, DIMENSIONS, DIMENSIONS//SIZE):
        x_st = 0
        x_end = DIMENSIONS

        draw.line(((x_st, y), (x_end, y)), fill=0)

    # text
    font = ImageFont.truetype(FONT_TFF, FONT_SIZE)

    row = 0
    coloum = 0

    for x in range((DIMENSIONS//SIZE)//2, DIMENSIONS, DIMENSIONS//SIZE):
        for y in range((DIMENSIONS//SIZE)//2, DIMENSIONS, DIMENSIONS//SIZE):
            text = str(array[coloum][row]) + " " + unit  # reversed as the vals are entered as following statement
            centre_offset = FONT_SIZE  # kinda centres the text, can workout the exact thing probably but unnecessary
            draw.text((x-centre_offset, y), text, font=font, fill=0)
            coloum += 1
        row += 1
        coloum = 0

    del draw

    if __name__ == "__main__":
        img.show()
    else:
        img.save(os.path.join(path, "manual_result.jpg"))


# to test it
if __name__ == "__main__":
    temp = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]
    drawSquareGrid("1", temp, "K")

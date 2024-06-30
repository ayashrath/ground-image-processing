# Ground Image Processing

## Aim

The objective is to detect temperatures from square grids that represent earth regions in the testing condtions (which was to be represented by a 5x5 square grid with each cell having dedicated heating capability), with the intent to predict wildfires.

## Info on Project

This project contains 2 different versions, as the specifications of the testing room changed (specified below).

### V1

Initially the plan was to use 2 different cameras, one infrared camera and one normal camera. The normal camera detects the regions (the squares in the square grid), which is used onto the infrared image (while accounting for the relative position difference of the cameras).

The code is partially complete, and only contains the region detection and validation code.

### V2

The testing condition details were later specified to not have a light source, thus making the normal camera inoperable. Thus approch had to change. This is the current iteration, and is under developement.

The current developement info

1. The thermal image from the camera is sent from the CubeSat via UDP
   1. The temperatures are in Kelvin
2. The image is stored onto the "Ground Machine" in the following format
   1. Each instance is stored in a dir with name based on unix time of creation
   2. The dir holds 3 files - one for the unit of measurement, one for the raw temp data and one for the image file
   3. The reason is that the image file would need the data to be normalised, thus it could create issues during extraction

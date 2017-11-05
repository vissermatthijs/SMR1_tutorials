import argparse
import numpy as np

import cv2
import plantcv as pcv


def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=False)
    args = parser.parse_args()
    return args


args = options()
debug = args.debug
device = 0
img = cv2.imread("/home/matthijs/Downloads/tomaat_3.jpg")
img_2 = cv2.imread("/home/matthijs/Downloads/tomaat_3.jpg", 0)

cimg = cv2.cvtColor(img_2, cv2.COLOR_BAYER_BG2BGR)
device, s = pcv.rgb2gray_hsv(img, 's', device)

device, a = pcv.rgb2gray_lab(img, 'a', device, debug)  # looks most promissing

device, a_thresh = pcv.binary_threshold(a, 135, 255, 'light', device, debug)

device, a_mblur = pcv.median_blur(a_thresh, 5, device, debug)
kernel = np.zeros((3, 3), dtype=np.uint8)
device, mask_watershed, = pcv.erode(a_mblur, 5, 1, device, debug)

circles = cv2.HoughCircles(img_2, cv2.cv.CV_HOUGH_GRADIENT, 1, 30, param1=50, param2=20, minRadius=130, maxRadius=220)
print("watt?")

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 255, 0), 3)
cv2.imshow("detect circels", cimg)
cv2.waitKey(2000000)

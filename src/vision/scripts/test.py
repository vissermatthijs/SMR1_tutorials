#!/usr/bin/python
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import plantcv as pcv

# test comment


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)


while(1):
    # Read image
    img, path, filename = pcv.readimage("yucca2.JPG")

    device = 0

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    # Convert RGB to HSV and extract the Saturation channel
    device, s = pcv.rgb2gray_hsv(img, 's', device)
    device, s_thresh = pcv.binary_threshold(s, 120, 255, 'light', device)
    # Median Filter
    device, s_mblur = pcv.median_blur(s_thresh, 5, device)
    device, s_cnt = pcv.median_blur(s_thresh, 5, device)

    device, b = pcv.rgb2gray_lab(img, 'b', device)

    # Threshold the blue image
    device, b_thresh = pcv.binary_threshold(b, 137, 255, 'light', device)
    device, b_cnt = pcv.binary_threshold(b, 140, 255, 'light', device)
    device, bs = pcv.logical_or(b_thresh, b_cnt, device)

    device, masked = pcv.apply_mask(img, bs, 'white', device)

    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    device, masked_a = pcv.rgb2gray_lab(masked, 'a', device)
    device, masked_b = pcv.rgb2gray_lab(masked, 'b', device)



    # Threshold the green-magenta and blue images
    device, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', device)
    device, maskeda_thresh1 = pcv.binary_threshold(masked_a, 135, 255, 'light', device)
    device, maskedb_thresh = pcv.binary_threshold(masked_b, 150, 255, 'light', device)

    # Join the thresholded saturation and blue-yellow images (OR)
    device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device)
    device, ab = pcv.logical_or(maskeda_thresh1, ab1, device)
    device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device)

    # Fill small objects
    device, ab_fill = pcv.fill(ab, ab_cnt, 200, device)

    # Apply mask (for vis images, mask_color=white)
    device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device)






    cv2.imshow('image', masked2)
    cv2.waitKey(1)

cv2.destroyAllWindows()
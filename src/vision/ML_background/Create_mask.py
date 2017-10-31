# Pipeline starts by importing necessary packages, and defining user inputs
import argparse
import glob
import numpy as np

import cv2
import plantcv as pcv


# test comment


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args


def back_for_ground_sub(img, sliders):
    args = options()
    debug = args.debug
    stop = 0
    sat_thresh = 90
    blue_thresh = 135
    green_magenta_dark_thresh = 124
    green_magenta_light_thresh = 180
    blue_yellow_thresh = 128

    def nothing(x):
        pass

    if sliders == True:
        Stop = np.zeros((100, 512, 3), np.uint8)
        cv2.namedWindow('Saturation', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Blue', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Green_magenta_dark', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Green_magenta_light', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Blue_yellow_light', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Stop')
        cv2.createTrackbar('sat_thresh', 'Saturation', 85, 255, nothing)
        cv2.createTrackbar('blue_thresh', 'Blue', 135, 255, nothing)
        cv2.createTrackbar('green_magenta_dark_thresh', 'Green_magenta_dark', 117, 255, nothing)
        cv2.createTrackbar('green_magenta_light_thresh', 'Green_magenta_light', 180, 255, nothing)
        cv2.createTrackbar('blue_yellow_thresh', 'Blue_yellow_light', 128, 255, nothing)
        cv2.createTrackbar('stop', 'Stop', 0, 1, nothing)
    while (stop == 0):

        if sliders == True:
            # get current positions of five trackbars
            sat_thresh = cv2.getTrackbarPos('sat_thresh', 'Saturation')
            blue_thresh = cv2.getTrackbarPos('blue_thresh', 'Blue')
            green_magenta_dark_thresh = cv2.getTrackbarPos('green_magenta_dark_thresh', 'Green_magenta_dark')
            green_magenta_light_thresh = cv2.getTrackbarPos('green_magenta_light_thresh', 'Green_magenta_light')
            blue_yellow_thresh = cv2.getTrackbarPos('blue_yellow_thresh', 'Blue_yellow_light')

        # Pipeline step
        device = 0
        # Convert RGB to HSV and extract the Saturation channel
        # Extract the light and dark form the image
        device, s = pcv.rgb2gray_hsv(img, 's', device)
        # device, s_thresh = pcv.binary_threshold(s, sat_thresh, 255, 'light', device)
        device, s_thresh = pcv.otsu_auto_threshold(s, 255, 'light', device, debug)
        device, s_mblur = pcv.median_blur(s_thresh, 5, device)
        device, s_cnt = pcv.median_blur(s_thresh, 5, device)

        # Convert RGB to LAB and extract the Blue channel
        # Threshold the blue image
        # Combine the threshed saturation and the blue theshed image with the logical or
        device, b = pcv.rgb2gray_lab(img, 'b', device)
        device, b_thresh = pcv.otsu_auto_threshold(b, 255, 'light', device, debug)
        device, b_cnt = pcv.otsu_auto_threshold(b, 255, 'light', device, debug)
        device, b_cnt_2 = pcv.binary_threshold(b, 135, 255, 'light', device, debug)

        device, bs = pcv.logical_or(s_mblur, b_cnt, device)
        # Mask the original image with the theshed combination of the blue&saturation
        device, masked = pcv.apply_mask(img, bs, 'white', device)

        # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
        device, masked_a = pcv.rgb2gray_lab(masked, 'a', device)
        device, masked_b = pcv.rgb2gray_lab(masked, 'b', device)

        # Focus on capturing the plant from the masked image 'masked'
        # Extract plant green-magenta and blue-yellow channels
        # Channels are threshold to cap different portions of the plant
        # Threshold the green-magenta and blue images
        # Images joined together
        # device, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', device)
        device, maskeda_thresh = pcv.binary_threshold(masked_a, green_magenta_dark_thresh, 255, 'dark', device,
                                                      debug)  # Original 115 New 125
        device, maskeda_thresh1 = pcv.binary_threshold(masked_a, green_magenta_light_thresh, 255, 'light',
                                                       device, debug)  # Original 135 New 170
        device, maskedb_thresh = pcv.binary_threshold(masked_b, blue_yellow_thresh, 255, 'light',
                                                      device, debug)  # Original 150`, New 165
        device, maskeda_thresh2 = pcv.binary_threshold(masked_a, green_magenta_dark_thresh, 255, 'dark',
                                                       device, debug)  # Original 115 New 125

        # Join the thresholded saturation and blue-yellow images (OR)
        device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device, debug)
        device, ab = pcv.logical_or(maskeda_thresh1, ab1, device, debug)
        device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device, debug)
        device, ab_cnt_2 = pcv.logical_and(b_cnt_2, maskeda_thresh2, device, debug)
        # Fill small objects
        device, ab_fill = pcv.fill(ab, ab_cnt, 200, device, debug)  # Original 200 New: 120
        # cv2.imwrite("yucca_1.jpg",s)
        device, mask_new = pcv.logical_and(maskeda_thresh2, maskedb_thresh, device, debug)

        # Apply mask (for vis images, mask_color=white)
        device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device, debug)
        device, masked3 = pcv.apply_mask(masked, ab_cnt_2, 'white', device, debug)
        # Identify objects
        device, id_objects, obj_hierarchy = pcv.find_objects(masked2, ab_fill, device, debug)
        # Define ROI

        # Plant extracton done-----------------------------------------------------------------------------------


        if sliders == True:
            stop = cv2.getTrackbarPos('stop', 'Stop')
            cv2.imshow('Stop', Stop)
            cv2.imshow('Saturation', s_thresh)
            cv2.imshow('Blue', b_thresh)
            cv2.imshow('Green_magenta_dark', maskeda_thresh)
            cv2.imshow('Green_magenta_light', maskeda_thresh1)
            cv2.imshow('Blue_yellow_light', maskedb_thresh)
            cv2.imshow('Mask', mask_new)
            # cv2.imshow('Mask', masked)
            # cv2.imshow('Mask2', masked2)
            # cv2.imshow('Mask3', masked3)
            # cv2.imshow('masked_a', masked_a)
            # cv2.imshow('masked_b', masked_b)
            # cv2.imshow('fill', ab_fill)
            # cv2.imshow('ab_cnt', ab)
            # cv2.imshow('ab1', ab1)
            # cv2.imshow('ab_cnt2', ab_cnt_2)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        else:
            stop = 1

    device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, roi=None, roi_input='default',
                                                 debug=False, adjust=True, x_adj=100, y_adj=50, w_adj=-150,
                                                 h_adj=-50)

    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device,
                                                                           debug=False)

    # Object combine kept objects
    device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device, debug=False)
    return device, ab_fill, s_thresh, obj


cv_img = []
i = 0
for img in glob.glob("Data_set/Foto's/*.jpg"):
    n = cv2.imread(img)
    i = i + 1
    # Read image
    # img = cv2.imread("yucca_1.jpg")
    # img, path, filename = pcv.readimage("Data_set/Original/7.jpg")
    # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    device, ab_fill, masked2, obj = back_for_ground_sub(n, True)
    print("test")
    cv2.imwrite("Data_set/Mask/" + str(i) + ".png", masked2)
    cv2.imwrite("Data_set/Original/" + str(i) + ".png",n)

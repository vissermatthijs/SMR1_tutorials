#!/usr/bin/python

import argparse
import numpy as np

# Pipeline starts by importing necessary packages, and defining user inputs
import cv2
import plantcv as pcv


# test comment


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=True)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.")
    args = parser.parse_args()
    return args


def back_for_ground_sub(img, sliders):
    args = options()
    debug = args.debug
    stop = 0
    sat_thresh = 85
    blue_thresh = 135
    green_magenta_dark_thresh = 117
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
        device, s_thresh = pcv.binary_threshold(s, sat_thresh, 255, 'light', device)
        device, s_mblur = pcv.median_blur(s_thresh, 5, device)
        device, s_cnt = pcv.median_blur(s_thresh, 5, device)

        # Convert RGB to LAB and extract the Blue channel
        # Threshold the blue image
        # Combine the threshed saturation and the blue theshed image with the logical or
        device, b = pcv.rgb2gray_lab(img, 'b', device)
        device, b_thresh = pcv.binary_threshold(b, blue_thresh, 255, 'light', device)
        device, b_cnt = pcv.binary_threshold(b, blue_thresh, 255, 'light', device)
        device, b_cnt_2 = pcv.binary_threshold(b, 135, 255, 'light', device)

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
                                                      debug='print')  # Original 115 New 125
        device, maskeda_thresh1 = pcv.binary_threshold(masked_a, green_magenta_light_thresh, 255, 'light',
                                                       device)  # Original 135 New 170
        device, maskedb_thresh = pcv.binary_threshold(masked_b, blue_yellow_thresh, 255, 'light',
                                                      device)  # Original 150`, New 165
        device, maskeda_thresh2 = pcv.binary_threshold(masked_a, green_magenta_dark_thresh, 255, 'dark',
                                                       device)  # Original 115 New 125

        # Join the thresholded saturation and blue-yellow images (OR)
        device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device)
        device, ab = pcv.logical_or(maskeda_thresh1, ab1, device)
        device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device)
        device, ab_cnt_2 = pcv.logical_and(b_cnt_2, maskeda_thresh2, device)
        # Fill small objects
        device, ab_fill = pcv.fill(ab, ab_cnt, 200, device)  # Original 200 New: 120

        # Apply mask (for vis images, mask_color=white)
        device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device)
        device, masked3 = pcv.apply_mask(masked, ab_cnt_2, 'white', device)
        # Identify objects
        device, id_objects, obj_hierarchy = pcv.find_objects(masked2, ab_fill, device)
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
            cv2.imshow('Mask', masked)
            cv2.imshow('Mask2', masked2)
            cv2.imshow('Mask3', masked3)
            cv2.imshow('masked_a', masked_a)
            cv2.imshow('masked_b', masked_b)
            cv2.imshow('fill', ab_fill)
            cv2.imshow('ab_cnt', ab)
            cv2.imshow('ab1', ab1)
            cv2.imshow('ab_cnt2', ab_cnt_2)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        else:
            stop = 1
    device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, roi=None, roi_input='default',
                                                 debug="plot", adjust=True, x_adj=100, y_adj=50, w_adj=-150,
                                                 h_adj=-50)

    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device,
                                                                           debug="plot")

    # Object combine kept objects
    device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device, debug="plot")
    return device, ab_fill, mask, obj


# Read image
img, path, filename = pcv.readimage("yucca3.jpg")
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

device, ab_fill, masked2,obj = back_for_ground_sub(img, False)

# Segment image with watershed function
device, watershed_header, watershed_data,analysis_images=pcv.watershed_segmentation(device, img,masked2,10,'./examples',debug='plot')

print(watershed_header)
print(watershed_data)
# Find shape properties, output shape image (optional)
# device, shape_header, shape_data, shape_img = pcv.analyze_object(img, args.image, obj, mask, device,False)

# Shape properties relative to user boundary line (optional)
# device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(img, args.image, obj, mask, 1680, False)

# Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
# device, color_header, color_data, color_img = pcv.analyze_color(img, args.image, kept_mask, 256, device, False,'all', 'v', 'img', 300,False)
# plt.plot(shape_img)
# plt.show()
# cv2.imshow('shape',shape_img)
# cv2.imshow('color',color_img)
# cv2.imshow('boundry',boundary_img1)

# Find shape properties, output shape image (optional)
device, shape_header, shape_data, shape_img = pcv.analyze_object(img, "Yucca", obj, masked2, device, debug = "plot")

# Shape properties relative to user boundary line (optional)
device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(img, "Yucca", obj, masked2, 1680, device, debug = "plot")

# Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
#device, color_header, color_data, color_img = pcv.analyze_color(img, "Yucca", ma, 256, device, debug = "plot", 'all', 'v', 'img', 300, args.outdir + '/' + filename)

# Starting skeletoning----------------------------------------------------
print(
"Plant extracton done-----------------------------------------------------------------------------------Starting skeletoning")
size = np.size(masked2)

skel = np.zeros(masked2.shape, np.uint8)

ret, mask_thresh = cv2.threshold(masked2, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
print(element)

done = False
print(size)
while (not done):

    eroded = cv2.erode(mask_thresh, element)

    temp = cv2.dilate(eroded, element)

    temp = cv2.subtract(mask_thresh, temp)
    skel = cv2.bitwise_or(skel, temp)
    mask_thresh = eroded.copy()

    zeros = size - cv2.countNonZero(mask_thresh)
    print(zeros)
    if zeros == size:
        done = True
print("-------------------------------Done makeing skelinton------------------------------")

"""
#-----------------Line segments
minLineLength=100
lines = cv2.HoughLinesP(image=skel,rho=1.5,theta=np.pi/360, threshold=80,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)

print(lines)
a,b,c = lines.shape

for x1,y1,x2,y2 in lines[0]:
    cv2.line(masked2,(x1,y1),(x2,y2),(0,0,255),2)
"""

lines = cv2.HoughLines(image=skel, rho=1, theta=np.pi / 180, threshold=100, srn=0, stn=0)

for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

print("---------------Visualizing everything-----")
cv2.imshow('mask', masked2)
cv2.imshow('image', img)
cv2.imshow("skel", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()

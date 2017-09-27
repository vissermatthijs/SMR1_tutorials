#!/usr/bin/python

#Pipeline starts by importing necessary packages, and defining user inputs
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
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args


# Read image
img, path, filename = pcv.readimage("yucca2.JPG")
args = options()


def back_for_ground_sub(img,sliders):
    stop = 0
    def nothing(x):
        pass

    if sliders == True:
        Saturation = np.zeros((300, 512, 3), np.uint8)
        Blue = np.zeros((300, 512, 3), np.uint8)
        Green_magenta_dark = np.zeros((300, 512, 3), np.uint8)
        Green_magenta_light = np.zeros((300, 512, 3), np.uint8)
        Blue_yellow_light = np.zeros((300, 512, 3), np.uint8)
        Stop_ = np.zeros((300, 512, 3), np.uint8)

        cv2.namedWindow('Saturation')
        cv2.namedWindow('Blue')
        cv2.namedWindow('Green_magenta_dark')
        cv2.namedWindow('Green_magenta_light')
        cv2.namedWindow('Blue_yellow_light')
        cv2.namedWindow('Stop')
        cv2.createTrackbar('sat_thresh', 'Saturation', 120, 255, nothing)
        cv2.createTrackbar('blue_thresh', 'Blue', 137, 255, nothing)
        cv2.createTrackbar('green_magenta_dark_thresh', 'Green_magenta_dark', 125, 255, nothing)
        cv2.createTrackbar('green_magenta_light_thresh', 'Green_magenta_light', 170, 255, nothing)
        cv2.createTrackbar('blue_yellow_thresh', 'Blue_yellow_light', 165, 255, nothing)
        cv2.createTrackbar('stop','Stop',0,1,nothing)
    while(stop == 0):
        if sliders == True:
            # get current positions of five trackbars
            sat_thresh = cv2.getTrackbarPos('sat_thresh', 'Saturation')
            blue_thresh = cv2.getTrackbarPos('blue_thresh','Blue')
            green_magenta_dark_thresh = cv2.getTrackbarPos('green_magenta_dark_tresh', 'Green_magenta_dark')
            green_magenta_light_thresh = cv2.getTrackbarPos('green_magenta_light_tresh', 'Green_magenta_light')
            blue_yellow_thresh = cv2.getTrackbarPos('blue_yellow_tresh', 'Blue_yellow_light')

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
        device, b_cnt = pcv.binary_threshold(b, 140, 255, 'light', device)
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
        #device, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', device)
        device, maskeda_thresh = pcv.binary_threshold(masked_a, green_magenta_dark_thresh, 255, 'dark', device) #Original 115 New 125
        device, maskeda_thresh1 = pcv.binary_threshold(masked_a, green_magenta_light_thresh, 255, 'light', device)#Original 135 New 170
        device, maskedb_thresh = pcv.binary_threshold(masked_b, blue_yellow_thresh, 255, 'light', device)# Original 150`, New 165

        # Join the thresholded saturation and blue-yellow images (OR)
        device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device)
        device, ab = pcv.logical_or(maskeda_thresh1, ab1, device)
        device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device)

        # Fill small objects
        device, ab_fill = pcv.fill(ab, ab_cnt, 120, device)#Original 200 New: 120

        # Apply mask (for vis images, mask_color=white)
        device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device)



        if sliders == True:
            stop = cv2.getTrackbarPos('stop', 'Stop')
            cv2.imshow('Saturion_tresh',Saturation)
            cv2.imshow('Blue_thresh',Blue)
            cv2.imshow('Green_magenta_dark',Green_magenta_dark)
            cv2.imshow('Green_magenta_light',Green_magenta_light)
            cv2.imshow('Yellow_blue',Blue_yellow_light)
        else:
            stop = 1
        print('1')
    return masked2, ab_fill, device
masked2, ab_fill, device = back_for_ground_sub(img, True)
# Identify objects
device, id_objects, obj_hierarchy = pcv.find_objects(masked2, ab_fill, device)
# Define ROI
device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, None, 'default',None, True, 100, 200,-300, -250)

# Decide which objects to keep
device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy,id_objects, obj_hierarchy, device)

# Object combine kept objects
device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device)

#Plant extracton done-----------------------------------------------------------------------------------

# Find shape properties, output shape image (optional)
device, shape_header, shape_data, shape_img = pcv.analyze_object(img, args.image, obj, mask, device,False)

# Shape properties relative to user boundary line (optional)
device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(img, args.image, obj, mask, 1680, False)

# Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
device, color_header, color_data, color_img = pcv.analyze_color(img, args.image, kept_mask, 256, device, False,'all', 'v', 'img', 300,False)

#Starting skeletoning----------------------------------------------------
print("Plant extracton done-----------------------------------------------------------------------------------Starting skeletoning")
size = np.size(mask)

skel = np.zeros(mask.shape,np.uint8)

ret,mask_thresh = cv2.threshold(mask,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
print(element)

done = False
print(size)
while( not done):

    eroded = cv2.erode(mask_thresh,element)

    temp = cv2.dilate(eroded,element)

    temp = cv2.subtract(mask_thresh,temp)
    skel = cv2.bitwise_or(skel,temp)
    mask_thresh = eroded.copy()

    zeros = size - cv2.countNonZero(mask_thresh)
    print(zeros)
    if zeros==size:
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

lines=cv2.HoughLines(image=skel,rho=1,theta=np.pi/180,threshold=100,srn=0,stn=0)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(masked2,(x1,y1),(x2,y2),(0,0,255),2)


print("---------------Visualizing everything-----")
cv2.imshow('mask', mask)
cv2.imshow('image', masked2)
cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()

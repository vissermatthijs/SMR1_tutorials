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

#cv2.namedWindow('image')
#cv2.createTrackbar('R','image',0,255,nothing)
#cv2.createTrackbar('G','image',0,255,nothing)
#cv2.createTrackbar('B','image',0,255,nothing)


# Read image
img, path, filename = pcv.readimage("yucca2.JPG")

device = 0

# get current positions of four trackbars
R = cv2.getTrackbarPos('R', 'image')
G = cv2.getTrackbarPos('G', 'image')
B = cv2.getTrackbarPos('B', 'image')
# Convert RGB to HSV and extract the Saturation channel
device, s = pcv.rgb2gray_hsv(img, 's', device)
device, s_thresh = pcv.binary_threshold(s, 120, 255, 'light', device)
#mine testdevice, s_thresh = pcv.binary_threshold(s, r, g, 'light', device)

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
#device, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', device)
#print(R,G,B)
device, maskeda_thresh = pcv.binary_threshold(masked_a, 125, 255, 'dark', device) #Original 115 New 125
device, maskeda_thresh1 = pcv.binary_threshold(masked_a, 170, 255, 'light', device)#Original 135 New 170
device, maskedb_thresh = pcv.binary_threshold(masked_b, 165, 255, 'light', device)# Original 150`, New 165

# Join the thresholded saturation and blue-yellow images (OR)
device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device)
device, ab = pcv.logical_or(maskeda_thresh1, ab1, device)
device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device)

# Fill small objects
device, ab_fill = pcv.fill(ab, ab_cnt, 120, device)#Original 200 New: 120

# Apply mask (for vis images, mask_color=white)
device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device)

# Identify objects
device, id_objects, obj_hierarchy = pcv.find_objects(masked2, ab_fill, device)
# Define ROI
device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, None, 'default',None, True, 100, 200,-300, -250)

# Decide which objects to keep
device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy,id_objects, obj_hierarchy, device)

# Object combine kept objects
device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device)

#Plant extracton done-----------------------------------------------------------------------------------
#Starting skeletoning----------------------------------------------------
print("Plant extracton done-----------------------------------------------------------------------------------Starting skeletoning")
size = np.size(mask)

skel = np.zeros(mask.shape,np.uint8)

ret,mask_thresh = cv2.threshold(mask,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

kernel = np.ones((5,5),np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
erosion = cv2.erode(mask_thresh,kernel,iterations = 1)

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
cv2.imshow("erosion",erosion)
cv2.imshow("tophat",tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

#!/usr/bin/python
import argparse
import numpy as np

import cv2
import plantcv as pcv


### Parse command-line argumentss
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-r", "--result", help="result file.", required=True)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default='print')
    args = parser.parse_args()
    return args


### Main pipeline
def main():
    print("step one")
    """
    Step one: Background forground substraction 
    """
    # Get options
    args = options()
    debug = args.debug
    # Read image
    img, path, filename = pcv.readimage(args.image)
    # Pipeline step
    device = 0
    device, resize_img = pcv.resize(img, 0.4, 0.4, device, debug)
    # Classify the pixels as plant or background
    device, mask_img = pcv.naive_bayes_classifier(resize_img,
                                                  pdf_file="/home/matthijs/PycharmProjects/SMR1/src/vision/ML_background/Trained_models/model_3/naive_bayes_pdfs.txt",
                                                  device=0, debug='print')

    # Median Filter
    device, blur = pcv.median_blur(mask_img.get('plant'), 5, device, debug)
    print("step two")
    """
    Step one: Identifiy the objects, extract and filter the objects
    """

    # Identify objects
    device, id_objects, obj_hierarchy = pcv.find_objects(resize_img, blur, device, debug=None)

    # Define ROI
    device, roi1, roi_hierarchy = pcv.define_roi(resize_img, 'rectangle', device, roi=True, roi_input='default',
                                                 debug=True, adjust=True, x_adj=50, y_adj=10, w_adj=-100,
                                                 h_adj=0)
    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(resize_img, 'cutto', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device, debug)
    # print(roi_objects[0])
    # cv2.drawContours(resize_img, [roi_objects[0]], 0, (0, 255, 0), 3)
    # cv2.imshow("img",resize_img)
    # cv2.waitKey(0)
    area_oud = 0
    i = 0
    index = 0
    object_list = []
    # a = np.array([[hierarchy3[0][0]]])
    hierarchy = []
    for cnt in roi_objects:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m10"] or M["m01"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # check if the location of the contour is between the constrains
            if cX > 200 and cX < 500 and cY > 25 and cY < 400:
                # cv2.circle(resize_img, (cX, cY), 5, (255, 0, 255), thickness=1, lineType=1, shift=0)
            # check if the size of the contour is bigger than 250
                if area > 450:
                    obj = np.vstack(roi_objects)
                    object_list.append(roi_objects[i])
                    hierarchy.append(hierarchy3[0][i])
                    print(i)
        i = i + 1
    a = np.array([hierarchy])
    #a = [[[-1,-1,-1,-1][-1,-1,-1,-1][-1,-1,-1,-1]]]
    # Object combine kept objects
    # device, obj, mask_2 = pcv.object_composition(resize_img, object_list, a, device, debug)

    mask_contours = np.zeros(resize_img.shape, np.uint8)
    cv2.drawContours(mask_contours, object_list, -1, (255, 255, 255), -1)
    gray_image = cv2.cvtColor(mask_contours, cv2.COLOR_BGR2GRAY)
    ret, mask_contours = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Identify objects
    device, id_objects, obj_hierarchy = pcv.find_objects(resize_img, mask_contours, device, debug=None)
    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(resize_img, 'cutto', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device,
                                                                           debug=None)
    # Object combine kept objects
    device, obj, mask = pcv.object_composition(resize_img, roi_objects, hierarchy3, device, debug=None)
    ############### Analysis ################
    outfile = False
    if args.writeimg == True:
        outfile = args.outdir + "/" + filename

    print("step three")
    """
    Step three: Calculate all the features
    """
    # Find shape properties, output shape image (optional)
    device, shape_header, shape_data, shape_img = pcv.analyze_object(resize_img, args.image, obj, mask, device, debug,
                                                                     args.outdir + '/' + filename)

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(resize_img, args.image, obj, mask, 1680,
                                                                              device,
                                                                              debug, args.outdir + '/' + filename)

    # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(resize_img, args.image, kept_mask, 256, device,
                                                                    debug,
                                                                    'all', 'v', 'img', 300,
                                                                    args.outdir + '/' + filename)
    device, watershed_header, watershed_data, analysis_images = pcv.watershed_segmentation(device, resize_img, mask, 50,
                                                                                           './examples', debug)
    device, list_of_acute_points = pcv.acute_vertex(obj, 30, 60, 10, resize_img, device, debug)

    device, top, bottom, center_v = pcv.x_axis_pseudolandmarks(obj, mask, resize_img, device, debug)

    device, left, right, center_h = pcv.y_axis_pseudolandmarks(obj, mask, resize_img, device, debug)

    device, points_rescaled, centroid_rescaled, bottomline_rescaled = pcv.scale_features(obj, mask,
                                                                                         list_of_acute_points,
                                                                                         225, device, debug)
    # Identify acute vertices (tip points) of an object
    # Results in set of point values that may indicate tip points
    device, vert_ave_c, hori_ave_c, euc_ave_c, ang_ave_c, vert_ave_b, hori_ave_b, euc_ave_b, ang_ave_b = pcv.landmark_reference_pt_dist(
        points_rescaled, centroid_rescaled, bottomline_rescaled, device, debug)
    # Write shape and color data to results fil
    result = open(args.result, "a")
    result.write('\t'.join(map(str, points_rescaled)))
    result.write("\n")
    result.write('\t'.join(map(str, shape_header)))
    result.write("\n")
    result.write('\t'.join(map(str, shape_data)))
    result.write("\n")
    for row in shape_img:
        result.write('\t'.join(map(str, row)))
        result.write("\n")
    result.write('\t'.join(map(str, color_header)))
    result.write("\n")
    result.write('\t'.join(map(str, color_data)))
    result.write("\n")
    for row in color_img:
        result.write('\t'.join(map(str, row)))
        result.write("\n")
    result.close()
    print("done")

if __name__ == '__main__':
    main()

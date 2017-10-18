# !/usr/bin/python
# !/usr/bin/python
import argparse
import csv
import numpy as np

import cv2
import plantcv as pcv
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


### Parse command-line argumentss
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=True)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default='print')
    args = parser.parse_args()
    return args


### Main pipeline
def get_feature(img):
    print("step one")
    """
    Step one: Background forground substraction 
    """
    # Get options
    args = options()
    debug = args.debug
    # Read image
    filename = args.result
    # img, path, filename = pcv.readimage(args.image)
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
    # a = [[[-1,-1,-1,-1][-1,-1,-1,-1][-1,-1,-1,-1]]]
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
    device, shape_header, shape_data, shape_img = pcv.analyze_object(resize_img, args.image, obj, mask, device, debug
                                                                     )

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(resize_img, args.image, obj, mask, 1680,
                                                                              device
                                                                              )

    # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(resize_img, args.image, kept_mask, 256, device,
                                                                    debug,
                                                                    'all', 'v', 'img', 300
                                                                    )

    maks_watershed = mask.copy()
    kernel = np.zeros((5, 5), dtype=np.uint8)
    device, mask_watershed, = pcv.erode(maks_watershed, 5, 1, device, debug)

    device, watershed_header, watershed_data, analysis_images = pcv.watershed_segmentation(device, resize_img, mask, 50,
                                                                                           './examples', debug)
    device, list_of_acute_points = pcv.acute_vertex(obj, 30, 60, 10, resize_img, device, debug)

    device, top, bottom, center_v = pcv.x_axis_pseudolandmarks(obj, mask, resize_img, device, debug)

    device, left, right, center_h = pcv.y_axis_pseudolandmarks(obj, mask, resize_img, device, debug)

    device, points_rescaled, centroid_rescaled, bottomline_rescaled = pcv.scale_features(obj, mask,
                                                                                         list_of_acute_points, 225,
                                                                                         device, debug)

    # Identify acute vertices (tip points) of an object
    # Results in set of point values that may indicate tip points
    device, vert_ave_c, hori_ave_c, euc_ave_c, ang_ave_c, vert_ave_b, hori_ave_b, euc_ave_b, ang_ave_b = pcv.landmark_reference_pt_dist(
        points_rescaled, centroid_rescaled, bottomline_rescaled, device, debug)

    landmark_header = ['HEADER_LANDMARK', 'tip_points', 'tip_points_r', 'centroid_r', 'baseline_r', 'tip_number',
                       'vert_ave_c',
                       'hori_ave_c', 'euc_ave_c', 'ang_ave_c', 'vert_ave_b', 'hori_ave_b', 'euc_ave_b', 'ang_ave_b',
                       'left_lmk', 'right_lmk', 'center_h_lmk', 'left_lmk_r', 'right_lmk_r', 'center_h_lmk_r',
                       'top_lmk', 'bottom_lmk', 'center_v_lmk', 'top_lmk_r', 'bottom_lmk_r', 'center_v_lmk_r']
    landmark_data = ['LANDMARK_DATA', 0, 0, 0, 0, len(list_of_acute_points), vert_ave_c,
                     hori_ave_c, euc_ave_c, ang_ave_c, vert_ave_b, hori_ave_b, euc_ave_b, ang_ave_b, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0]
    shape_data_train = list(shape_data)
    shape_data_train.pop(0)
    shape_data_train.pop(10)
    watershed_data_train = list(watershed_data)
    watershed_data_train.pop(0)
    landmark_data_train = [len(list_of_acute_points), vert_ave_c,
                           hori_ave_c, euc_ave_c, ang_ave_c, vert_ave_b, hori_ave_b, euc_ave_b, ang_ave_b]
    X = shape_data_train + watershed_data_train + landmark_data_train
    print("len X", len(X))
    print(X)
    # Write shape and color data to results fil
    result = open(args.result, "a")
    result.write('\t'.join(map(str, shape_header)))
    result.write("\n")
    result.write('\t'.join(map(str, shape_data)))
    result.write("\n")
    result.write('\t'.join(map(str, watershed_header)))
    result.write("\n")
    result.write('\t'.join(map(str, watershed_data)))
    result.write("\n")
    result.write('\t'.join(map(str, landmark_header)))
    result.write("\n")
    result.write('\t'.join(map(str, landmark_data)))
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
    return X
    print("done")


def train_model():
    # ____variables____
    print('training model....')
    seed = 5  # random_state
    test_size = 0.21  # test_size
    n_components = 3  # LDA components

    data = []
    target = []

    with open('plant_db.csv') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in dataset:
            data.append(row[1:])
            target.append(row[0])

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    # X_test_norm = scaler.transform(X_test)

    X_train_norm, X_test_norm, Y_train, Y_test = train_test_split(data, target, test_size=test_size, random_state=seed)
    print(X_test_norm[0:1])
    # print(data)


    # fit model no training datasad
    model = XGBClassifier(
        learning_rate=0.24,
        max_depth=8,
        n_estimators=15,
        silent=False,
        objective='binary:logistic',
        nthread=-1,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=0.5,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=2.6,
        reg_lambda=5,
        scale_pos_weight=1,
        base_score=0.5,
        seed=0
    )
    # train model with data
    model.fit(X_train_norm, Y_train)
    print('done')
    return model, scaler


print("opening cam")
# cap = cv2.VideoCapture(1)
# cap.set(3, 1920)
# cap.set(4, 1080)
model, scaler = train_model()

while True:
    # ret, frame = cap.read()
    # input_ = input("Cap image? :")
    # print(input_)
    frame = cv2.imread(
        "/home/matthijs/PycharmProjects/SMR1/src/vision/scripts/yucca_rename/yucca2/cam1_17-12-10_yucca2_223.png")
    input_ = "y"
    if str(input_) == "y":
        X = get_feature(frame)
        X = np.array(X)
        X = scaler.transform([X])
        y_pred = model.predict(X)
        print(y_pred)

    else:
        pass
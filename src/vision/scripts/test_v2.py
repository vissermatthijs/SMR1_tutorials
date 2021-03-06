import numpy as np
import sys

import cv2
import weave


def _thinningIteration(im, iter):
    I, M = im, np.zeros(im.shape, np.uint8)
    expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);
			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
					 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
					 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
					 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	} 
	"""

    weave.inline(expr, ["I", "iter", "M"])
    return (I & ~M)


def thinning(src):
    dst = src.copy() / 255
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
        dst = _thinningIteration(dst, 0)
        dst = _thinningIteration(dst, 1)
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break

    return dst * 255


def find_leaves_crosses(bw2):
    # start var:
    kernel = np.ones((10, 10), np.uint8)
    bw2_rgb = cv2.cvtColor(bw2, cv2.COLOR_GRAY2RGB)
    points = []
    rcs = []
    # step one: find points
    corners = cv2.goodFeaturesToTrack(bw2, maxCorners=300, qualityLevel=0.001, minDistance=30)
    kernel = np.ones((8, 8), np.uint8)
    ROI_scaled = cv2.dilate(bw2, kernel, iterations=5)
    _, bw2 = cv2.threshold(bw2, 127, 255, cv2.THRESH_BINARY)
    list = np.argwhere(bw2 == 255)
    # print("list",list)
    width, height = bw2.shape
    print("width", width)
    print("height", height)
    for points in list:
        # x, y = corner.ravel()
        y = points[0]
        x = points[1]
        print("x", x)
        print("y", y)

        ROI = bw2[int(y - 10):int(y + 10), int(x - 10):int(x + 10)]

        width, height = ROI.shape
        if width * height > 0:
            ROI_scaled = cv2.resize(ROI, (0, 0), fx=10, fy=10)

            ROI_scaled = cv2.dilate(ROI_scaled, kernel, iterations=5)

            ROI_scaled_border = cv2.copyMakeBorder(ROI_scaled, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                                   value=[255, 255, 255])

            ROI_inv = cv2.bitwise_not(ROI_scaled_border)

            _, ROI_inv = cv2.threshold(ROI_inv, 127, 255, cv2.THRESH_BINARY)

            ROI_scaled2 = ROI_inv.copy()

            cv2.imshow("INV", ROI_inv)

            contours_inv, hierarchy = cv2.findContours(ROI_inv, mode=cv2.RETR_LIST,
                                                       method=cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(ROI_scaled, mode=cv2.RETR_LIST,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)

            if len(contours_inv) > 2 and len(contours) == 1:
                cv2.circle(bw2_rgb, (x, y), 3, (0, 255, 0), -1)
                ROI_2 = bw2[int(y - 40):int(y + 40), int(x - 40):int(x + 40)]

                ROI_2 = cv2.dilate(ROI_2, kernel, iterations=1)

                ROI_scaled_border_2 = cv2.copyMakeBorder(ROI_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                                         value=[255, 255, 255])
                ROI_inv_2 = cv2.bitwise_not(ROI_scaled_border_2)

                _, ROI_inv_2 = cv2.threshold(ROI_inv_2, 127, 255, cv2.THRESH_BINARY)

                ROI_inv3 = ROI_inv_2.copy()

                contours_inv_2, hierarchy = cv2.findContours(ROI_inv_2, mode=cv2.RETR_LIST,
                                                             method=cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_inv_2) > 4:
                    cv2.circle(bw2_rgb, (int(x), int(y)), 3, (255, 0, 0), -1)
                print("contour_2", len(contours_inv_2))
                cv2.imshow("ROI_2", ROI_2)
                cv2.imshow("Roi_2", ROI_inv3)
                # cv2.waitKey(3000)
                # if len(contours_inv) > 3 and len(contours) == 1:
                # cv2.circle(bw2_rgb,(x, y), 3, (255,0,0), -1)

            ROI_rgb = cv2.cvtColor(ROI_scaled2, cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(ROI_rgb, contours_inv, -1, (0, 255, 0), 3)

            print(len(contours_inv))
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            # cv2.circle(bw2_rgb, (int(x),int(y)), 3, (255, 255, 0), -1)
    cv2.imshow('key_points', bw2_rgb)
    cv2.imshow("roi", ROI_rgb)
    cv2.imwrite("key_points.jpg", ROI_rgb)
    cv2.imwrite("key_points2.jpg", bw2_rgb)
    #cv2.waitKey(4000)






if __name__ == "__main__":
    src = cv2.imread("/home/matthijs/PycharmProjects/SMR1/src/vision/scripts/5_roi_mask.png")
    if src == None:
        sys.exit()
    bw = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    _, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
    bw2 = thinning(bw2)
    cv2.imshow("src", bw)
    minLineLength = bw2.shape[1] - 300
    cv2.imshow("star",bw2)
    lines = cv2.HoughLinesP(bw2, 1, np.pi / 180, 1, 10, 0.5)

    print(len(lines[0]))
    find_leaves_crosses(bw2)

    # for x1, y1, x2, y2 in lines[0]:
    # cv2.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # bw2, eigImage, tempImage, cornerCount, qualityLevel, minDistance, mask = None, blockSize = 3, useHarris = 0, k = 0.04


    cv2.imshow("feature_info", src)
    cv2.imwrite("src.jpg", src)
    cv2.imwrite("thinning.jpg", bw2)
    cv2.imwrite("info.jpg", src)
    cv2.imshow("thinning", bw2)
    cv2.waitKey()
    # calculate distance between all points
    # set a theshold fot the poits < max 100 pixels

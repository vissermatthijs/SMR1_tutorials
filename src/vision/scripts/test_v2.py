import numpy as np
import sys

import cv2
import shapely.geometry as shapgeo
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
    corners = cv2.goodFeaturesToTrack(bw2, maxCorners=300, qualityLevel=0.01, minDistance=20)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(bw2_rgb, (x, y), 3, (0, 255, 0), -1)
    cv2.imshow('key_points', bw2_rgb)
    cv2.waitKey(1000)
    # corners = np.int0(corners)

    # cv2.circle(bw2, (x, y), 3, 255, -1)
    # step two: create border for the points
    bw3 = cv2.dilate(bw2, kernel, iterations=4)
    # step three: find the contour and draw them
    cnts, hier = cv2.findContours(bw3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts_main = np.vstack(np.vstack(cnts))

    cnts_lines = shapgeo.asLineString(cnts_main)
    print(cnts_lines.length)
    print(len(corners))
    i = 0
    bw3_rgb = cv2.cvtColor(bw3, cv2.COLOR_GRAY2RGB)
    for corner in corners:
        x, y = corner.ravel()
        rcs = []
        for corner2 in corners:
            x2, y2 = corner2.ravel()
            corner_lines = shapgeo.LineString(((x, y), (x2, y2)))
            # print(corner_lines)
            # print(cnts_lines)
            if not (corner_lines.intersects(cnts_lines)):
                # print("Valid line found")
                i = i + 1
                dy = (y2 - y)
                dx = (x2 - x)
                if (dx != 0):
                    rc = (y2 - y) / (x2 - x)
                    angle = np.arctan(rc)
                    # print(angle)
                    rcs.append(angle)
                    cv2.line(bw3_rgb, (x, y), (x2, y2), (255, 0, 0))
                else:
                    rcs.append(1)
        # check if it is a corss point
        if len(rcs) > 0:
            if (np.sum(angle)) < 1:
                # print("draw poitn")
                cv2.circle(bw2_rgb, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(bw2_rgb, str(np.sum(np.round(angle, 2))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)

    print(i)
    cv2.drawContours(bw3_rgb, cnts, -1, (0, 255, 0), 1)
    cv2.imshow("bw3", bw3_rgb)
    cv2.imshow('crosses', bw2_rgb)
    # step three: draw line between points and check if it is between boundry






if __name__ == "__main__":
    src = cv2.imread("/home/matthijs/PycharmProjects/SMR1/src/vision/scripts/12_er_image_itr_1.png")
    if src == None:
        sys.exit()
    bw = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    _, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
    bw2 = thinning(bw2)
    cv2.imshow("src", bw)
    minLineLength = bw2.shape[1] - 300

    lines = cv2.HoughLinesP(bw2, 1, np.pi / 180, 1, 10, 0.5)

    print(len(lines[0]))
    find_leaves_crosses(bw2)

    # for x1, y1, x2, y2 in lines[0]:
    # cv2.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # bw2, eigImage, tempImage, cornerCount, qualityLevel, minDistance, mask = None, blockSize = 3, useHarris = 0, k = 0.04


    cv2.imshow("feature_info", src)
    cv2.imshow("thinning", bw2)
    cv2.waitKey()
    # calculate distance between all points
    # set a theshold fot the poits < max 100 pixels

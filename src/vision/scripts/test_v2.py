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


if __name__ == "__main__":
    src = cv2.imread("/home/matthijs/PycharmProjects/SMR1/src/vision/scripts/12_er_image_itr_1.png")
    if src == None:
        sys.exit()
    bw = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    _, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
    bw2 = thinning(bw2)
    cv2.imshow("src", bw)
    minLineLength = bw2.shape[1] - 300
    # lines = cv2.HoughLinesP(bw2, 1, np.pi / 180, 1, 100, 1)

    lines = cv2.HoughLinesP(image=bw2, rho=0.02, theta=np.pi / 500, threshold=10, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=100)

    a, b, c = lines.shape()
    for i in range(a):
        cv2.line(src, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

        # for x1, y1, x2, y2 in lines[0]:
        #    cv2.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # bw2, eigImage, tempImage, cornerCount, qualityLevel, minDistance, mask = None, blockSize = 3, useHarris = 0, k = 0.04
    corners = cv2.goodFeaturesToTrack(bw2, 100, 0.0001, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(src, (x, y), 3, 255, -1)

    cv2.imshow("feature_info", src)
    cv2.imshow("thinning", bw2)
    cv2.waitKey()
    # calculate distance between all points
    # set a theshold fot the poits < max 100 pixels

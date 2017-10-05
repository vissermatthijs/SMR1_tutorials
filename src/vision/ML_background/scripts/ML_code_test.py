import glob

import cv2
import plantcv as pcv

cv_img = []
for img in glob.glob("Data_set/Foto's/*.jpg"):
    n = cv2.imread(img)
    device = 0
    device, n = pcv.resize(n, 0.2, 0.2, device)
    # Classify the pixels as plant or background
    device, mask = pcv.naive_bayes_classifier(n, pdf_file="Trained_models/model_2/naive_bayes_pdfs.txt", device=0,
                                              debug=None)
    cv2.imshow("mask", mask.get('plant'))
    cv2.waitKey(2000)

import plantcv as pcv

# Read in a color image
img, path, filename = pcv.readimage("yucca_1.jpg")

# Classify the pixels as plant or background
device, mask = pcv.naive_bayes_classifier(img, pdf_file="naive_bayes_pdfs.txt", device=0, debug="plot")

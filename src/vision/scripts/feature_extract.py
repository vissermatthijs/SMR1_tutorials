#!/usr/bin/python
import argparse

import plantcv as pcv


### Parse command-line argumentss
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args


### Main pipeline
def main():
    # Get options
    args = options()
    debug = args.debug
    # Read image
    img, path, filename = pcv.readimage(args.image)

    # Pipeline step
    device = 0

    # Classify the pixels as plant or background
    device, mask = pcv.naive_bayes_classifier(img, pdf_file="naive_bayes_pdfs.txt", device=0, debug="plot")

    # Identify objects
    device, id_objects, obj_hierarchy = pcv.find_objects(img, mask, device, debug)

    # Define ROI
    device, roi1, roi_hierarchy = pcv.define_roi(mask, 'rectangle', device, None, 'default', debug, True, 550, 0,
                                                 -500, -1900)
    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device, debug)
    # Object combine kept objects
    device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device, debug)

    ############### Analysis ################

    outfile = False
    if args.writeimg == True:
        outfile = args.outdir + "/" + filename

    # Find shape properties, output shape image (optional)
    device, shape_header, shape_data, shape_img = pcv.analyze_object(img, args.image, obj, mask, device, debug,
                                                                     args.outdir + '/' + filename)

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(img, args.image, obj, mask, 1680, device,
                                                                              debug, args.outdir + '/' + filename)

    # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(img, args.image, kept_mask, 256, device, debug,
                                                                    'all', 'v', 'img', 300,
                                                                    args.outdir + '/' + filename)

    # Write shape and color data to results file
    result = open(args.result, "a")
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


if __name__ == '__main__':
    main()

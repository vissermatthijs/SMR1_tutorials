import glob

import cv2

# All file names should be in de format:
# camera-date-type-pic_number

yucca1 = 'cam1_17-12-10_yucca1_'
yucca2 = 'cam1_17-12-10_yucca2_'
yucca3 = 'cam1_17-12-10_yucca3_'
yuccu_dir = ("/home/matthijs/Plant_db/run5/yucca1/*.jpg", "/home/matthijs/Plant_db/run5/yucca2/*.jpg",
             "/home/matthijs/Plant_db/run5/yucca3/*.jpg")
count = 0
i = 0
for dir in yuccu_dir:
    count = count + 1
    print("processing catgorie" + str(count))
    for img in glob.glob(dir):
        image = cv2.imread(img)
        if count == 1:
            cv2.imwrite("yucca_rename/yucca1/" + yucca1 + str(i) + ".png", image)
        if count == 2:
            cv2.imwrite("yucca_rename/yucca2/" + yucca2 + str(i) + ".png", image)
        if count == 3:
            cv2.imwrite("yucca_rename/yucca3/" + yucca3 + str(i) + ".png", image)
            # print("yes")
        i = i + 1
print('done!')

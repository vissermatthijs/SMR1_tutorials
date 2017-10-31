import cv2
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)

cam = cv2.VideoCapture(2)
cv2.namedWindow("test")
img_type=""
img_counter = 0
waitkey = 1


while True:

    ret, frame = cam.read()

    if ser.readline() == b'1\r\n':
        img_name = str(img_counter)+".jpg"
        cv2.imwrite('yucca_all/'+img_name, frame)
        print("" + img_name + " written")
        img_counter += 1

cam.release()


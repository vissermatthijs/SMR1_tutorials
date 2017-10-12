import serial
import cv2
import os



ser = serial.Serial('/dev/cu.usbmodem1421', 9600)

cam = cv2.VideoCapture(1)
cv2.namedWindow("frame")
cv2.moveWindow("frame", 600, 300)

img_type=""
img_counter = 0

def show(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(0,0),fx= 0.35, fy=0.35)
    cv2.imshow("frame", gray)
    cv2.waitKey(1)

while True:


    if ser.readline() == b'1\r\n':
        ret, frame = cam.read()

        show(frame)
        wait  = 0
        img_type = ""

        while wait == 0 :
            img_type = input("Enter yucca type: ")
            wait = 1

        img_name =str(img_counter)+"-type-"+str(img_type)+".png"
        cv2.imwrite('yucca'+img_type+'/'+img_name, frame)
        print("" + img_name + " written")
        img_counter += 1
        ser.reset_input_buffer()

cam.release()
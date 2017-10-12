import serial
import cv2
import datetime
import os

ser = serial.Serial('/dev/cu.usbmodem1421', 9600)

cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)

cv2.namedWindow("frame")
cv2.moveWindow("frame", 600, 300)

img_type=""
img_counter = 0

def show(frame1):
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(0,0),fx= 0.35, fy=0.35)
    cv2.imshow("frame1", gray)
    cv2.waitKey(1)

while True:


    if ser.readline() == b'1\r\n':
        ret, frame1 = cam1.read()
        ret, frame2 = cam2.read()

        show(frame1)
        wait  = 0
        img_type = ""

        while wait == 0 :
            img_type = input("Enter yucca type: ")
            wait = 1

        dt = datetime.datetime.now().strftime('%m-%d_%H:%M')

        img_name ="cam1-"+"17-"+str(dt)+"_yucca"+str(img_type)+"_"+str(img_counter)+".png"
        cv2.imwrite('yucca'+img_type+'/'+img_name, frame1)
        print("" + img_name + " written")

        img2_name = "cam2-" + "17-" + str(dt) + "_yucca" + str(img_type) + "_" + str(img_counter) + ".png"
        cv2.imwrite('yucca' + img_type + '/' + img2_name, frame2)

        img_counter += 1
        ser.reset_input_buffer()

cam.release()
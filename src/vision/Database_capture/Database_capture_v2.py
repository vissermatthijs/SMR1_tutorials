import serial
import cv2
import datetime
import os


ser = serial.Serial('/dev/ttyACM0', 9600)


cv2.namedWindow("frame")
cv2.moveWindow("frame", 600, 300)

img_type=""
img_counter = 559

def show(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(0,0),fx= 0.35, fy=0.35)
    cv2.imshow("frame", gray)
    cv2.waitKey(1)

while True:


    if ser.readline() == b'1\r\n':

        frame = bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
        wait  = 0
        img_type = ""

        while wait == 0 :
            img_type = input("Enter yucca type: ")
            wait = 1

        if int(img_type) != 5:
            dt = datetime.datetime.now().strftime('%m-%d_%H:%M')

            img_name = "cam1-" + "17-" + str(dt) + "_yucca" + str(img_type) + "_" + str(img_counter) + ".png"
            cv2.imwrite('yucca' + img_type + '/' + img_name, frame)
            print("" + img_name + " written")
            img_counter += 1


        ser.reset_input_buffer()

cam.release()
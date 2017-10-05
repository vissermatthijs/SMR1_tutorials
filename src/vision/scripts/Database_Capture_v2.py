import cv2
import ids
import serial
import os

#sudo /etc/init.d/ueyeusbdrc stop
#sudo /etc/init.d/ueyeusbdrc start
#ueyesetid

cam = ids.Camera(1)
cam.color_mode = ids.ids_core.COLOR_RGB8
cam.exposure = 5
cam.auto_exposure = True
cam.continuous_capture = True

ser = serial.Serial('/dev/ttyACM0/', 9600)

while True:


    if ser.readline() == b'1\r\n':
        cv2.destroyAllWindows()
        img_start, meta = cam.next()
        img = cv2.cvtColor(img_start, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)


        img_name = "yucca_"+str(img_counter)+"_type_"+str(img_type)+".jpg"
        cv2.imwrite('yucca'+img_type+'/'+img_name, frame)

        print("" + img_name + " written")
        img_counter += 1

    else:
        print("error receiving image")

    if cv2.waitKey(1) == 27:
        cv2.imwrite('plant.jpg', img)
        break



cam.release()
cv2.destroyAllWindows()

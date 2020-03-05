import cv2 as cv
import numpy as np
import serial
import time

Arduino = serial.Serial('com8',9600) #create Serial Port arduino Serial Data
time.sleep(2) #wait for 2 seconds for the communication to get established

cam = cv.VideoCapture(0)

lower_red = np.array([0,125,125])#([20,100,100])
upper_red = np.array([10,225,225])#([40,255,255])

while(1):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)

    w = frame.shape[1]
    h = frame.shape[0]

    #smoothen the image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    # Define ROI
    mask = np.zeros_like(frame)
    mask[50:350, 50:350] = [225,225,225]

    image_roi = cv.bitwise_and(image_smooth, mask)
    cv.rectangle(frame, (50,50),(350,350),(0,0,255), 2)
    cv.line(frame,(150,50),(150,350),(0,0,255),1)
    cv.line(frame,(250,50),(250,350),(0,0,255),1)
    cv.line(frame,(50,150),(350,150),(0,0,255),1)
    cv.line(frame,(50,250),(350,250),(0,0,255),1)

    #Threshold the image for red colour
    img_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(img_hsv, lower_red, upper_red)

    #Find contours
    contours, heirarchy = cv.findContours(image_threshold, \
                                                         cv.RETR_TREE, \
                                                         cv.CHAIN_APPROX_NONE)



    #Find the index of largest contour
    if (len(contours)!=0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        #Pointer on video
        M = cv.moments(cnt)
        if(M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cx = int(M['m01']/M['m00'])
            cv.circle(frame, (cx, cy), 4, (0,255,0), -1)

            #cursor motion
            if cx in range(150,250):
                if cy < 150:
                    Arduino.write(b'w')
                    print("Fordward")

                    
                elif cy > 250:
                    Arduino.write(b's')
                    print("Backward")
                    
            
                else:
                    Arduino.write(b'q')
                    print("Stop")

            if cy in range(150,250):                                   
                if cx < 150:
                    Arduino.write(b'a')
                    print("Left")

                    
                elif cx > 250:
                    Arduino.write(b'd')
                    print("Right")
                    
            
                else:
                    Arduino.write(b'q')
                    print("Stop")       


    cv.imshow('Frame', frame)

    key = cv.waitkey(100)
    if key == 27:
        break

    cv.destroyAllWindows()        
            
                    
        

    
    
    

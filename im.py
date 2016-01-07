
import sys
import numpy as np
import cv2

def find_marker(image):

    #h,w, channels = img.shape

    #get red and sat
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    hue, sat, val = cv2.split(image)


    #find the marker by looking for red, with high saturation
    _,red = cv2.threshold(red, 128, 255, cv2.THRESH_BINARY)
    _,sat = cv2.threshold(sat, 128, 255, cv2.THRESH_BINARY)

    #AND the two thresholds, finding the car
    car = cv2.multiply(red, sat)

    #remove noise, highlighting the car
    #car = cv2.erode(car,car, iterations=5)
    #car = cv2.dilate(car,car,  iterations=5)

    obj,_,_ = cv2.findContours(car, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #import ipdb; ipdb.set_trace()

    return cv2.boundingRect(obj)

img_name = sys.argv[1]
img = cv2.imread(img_name,cv2.IMREAD_COLOR)
marker = find_marker(img)
color = (255,0,0)
cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], marker[1] + marker[3]),color)

cv2.imshow('pickchur',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

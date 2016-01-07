
import sys
import numpy as np
import cv2


def find_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of  color in HSV
    lower_c = np.array([0,50,50])
    upper_c = np.array([20,255,255])

    # Threshold the HSV image to get only desired colors
    mask = cv2.inRange(hsv, lower_c, upper_c)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask,res
    


def find_marker(image):

    #h,w, channels = img.shape

    #get red and sat
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    hue, sat, val = cv2.split(image)


    #find the marker by looking for red, with high saturation
    _,red = cv2.threshold(red, 128, 160, cv2.THRESH_BINARY)
    _,sat = cv2.threshold(sat, 128, 255, cv2.THRESH_BINARY)

    #AND the two thresholds, finding the car
    car = cv2.multiply(red, sat)

    #remove noise, highlighting the car
    elem = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    car = cv2.erode(car,elem, iterations=5)
    car = cv2.dilate(car,elem, iterations=5)
    #return cv2.boundingRect(car)

    img, contours,hierarchy = cv2.findContours(car.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #import ipdb; ipdb.set_trace()

    return cv2.boundingRect(contours[1])

img_name = sys.argv[1]
mode = int(sys.argv[2])
img = cv2.imread(img_name,cv2.IMREAD_COLOR)
if mode==1:
    marker = find_marker(img)
    #print marker
    color = (255,0,0)
    cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], marker[1] + marker[3]),color)
else:
    mask, res = find_color(img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

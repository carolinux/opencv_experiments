import os
import sys

import numpy as np
import cv2

#
#ffprobe ./Test3_Tr1_Session5.MOV -v 0 -select_streams v   -print_format flat -show_entries stream=r_frame_rate
# for tests on 10 Jan 2016
fps= 30000/1001
red_thres=[110,170]
green_thres=[0,60]
sat_thres=[80,240]
FRAME_EXT='jpg'


# turn vid to frames
#ffmpeg -i ./Test3_Tr1_Session5.MOV -s hd720 -r 30 -f image2  Test3_Tr1_Session5/image%05d.jpg


def find_marker(image, red_thres, green_thres, sat_thres):

    #h,w, channels = img.shape

    #get red and sat
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    hue, sat, val = cv2.split(hsv)
    #cv2.imshow('red',red)
    #cv2.imshow('green',green)
    #cv2.imshow('sat',sat)
    #cv2.imshow('blue',blue)
    #cv2.imshow('val',val)

    #find the marker by looking for red, with high saturation
    sat = cv2.inRange(sat, np.array((sat_thres[0])), np.array((sat_thres[1])))
    red = cv2.inRange(red, np.array((red_thres[0])), np.array((red_thres[1])))
    green = cv2.inRange(green, np.array((green_thres[0])), np.array((green_thres[1])))
    #_,red = cv2.threshold(red, red_thres[0], red_thres[1], cv2.THRESH_BINARY)
    #cv2.imshow('red_thres',red)

    #_,green = cv2.threshold(green, green_thres[0], green_thres[1], cv2.THRESH_BINARY)
    #_,sat = cv2.threshold(sat, sat_thres[0], sat_thres[1], cv2.THRESH_BINARY)
    #cv2.imshow('sat_thres',sat)
    #cv2.imshow('green_thres',green)
    #AND the two thresholds, finding the car
    car = cv2.multiply(red, sat)
    car = cv2.multiply(car, green)
    #cv2.imshow('mask',car)


    #remove noise, highlighting the car
    #elem = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #car = cv2.erode(car,elem, iterations=1)
    #car = cv2.dilate(car,elem, iterations=3)
    #return cv2.boundingRect(car)

    img, contours,hierarchy = cv2.findContours(car.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #import ipdb; ipdb.set_trace()

    #return cv2.boundingRect(contours[1])
    return map(lambda x: cv2.boundingRect(x),contours)

def process_img(img_path):

    img = cv2.imread(img_path,cv2.IMREAD_COLOR)

    markers = find_marker(img,red_thres=red_thres,green_thres=green_thres, sat_thres=sat_thres)
    color = (255,0,0)
    for marker in markers:
        print marker
        cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], marker[1] + marker[3]),color)


    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    folder = sys.argv[1]
    pics = os.listdir(folder) # also sort? 
    for pic in pics:
        if pic.endswith(FRAME_EXT):
            image_path = os.path.join(folder,pic)
            process_img(image_path)

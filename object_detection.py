import math
import os
import sys

import numpy as np
import cv2

#ffprobe ./Test3_Tr1_Session5.MOV -v 0 -select_streams v   -print_format flat -show_entries stream=r_frame_rate
# for tests on 10 Jan 2016
fps= 30000/1001
red_thres=[110,170]
green_thres=[0,60]
sat_thres=[80,240]
FRAME_EXT='png'
MAX_DIST=20 # max dist in pixels between two positions
pixel_to_meters_ratio=100/3.0


# turn vid to frames
#ffmpeg -i ./Test3_Tr1_Session5.MOV -s hd720 -r 30 -f image2  Test3_Tr1_Session5/image%05d.jpg


def find_marker(image, red_thres, green_thres, sat_thres):

    #h,w, channels = img.shape

    #get red and sat
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    hue, sat, val = cv2.split(hsv)

    #find the marker by looking for red, with high saturation
    sat = cv2.inRange(sat, np.array((sat_thres[0])), np.array((sat_thres[1])))
    red = cv2.inRange(red, np.array((red_thres[0])), np.array((red_thres[1])))
    green = cv2.inRange(green, np.array((green_thres[0])), np.array((green_thres[1])))
    #AND the two thresholds, finding the car
    car = cv2.multiply(red, sat)
    car = cv2.multiply(car, green)

    #remove noise (not doing it now because the POIs are very small)
    #elem = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #car = cv2.erode(car,elem, iterations=1)
    #car = cv2.dilate(car,elem, iterations=3)
    #return cv2.boundingRect(car)

    img, contours,hierarchy = cv2.findContours(car.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #import ipdb; ipdb.set_trace()

    #return cv2.boundingRect(contours[1])
    return map(lambda x: cv2.boundingRect(x),contours)

def center(marker):
    return (marker[0]+marker[2]/2.0, marker[1] + marker[3]/2.0)

def dist(p1,p2):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    return math.sqrt(dx*dx + dy*dy)

def process_img(img_path,show, prev_pos=None):

    img = cv2.imread(img_path,cv2.IMREAD_COLOR)

    markers = find_marker(img,red_thres=red_thres,green_thres=green_thres, sat_thres=sat_thres)
    correct_marker = None
    if prev_pos is None:
        if len(markers)!=1:
            raise Exception("Can't find initial position of beanie")
        curr_pos = center(markers[0])
        correct_marker = markers[0]

    else:
        found = False
        for marker in markers:
            if dist(center(marker),prev_pos)<MAX_DIST:
                curr_pos = center(marker)
                correct_marker = marker
                found=True

        if not found:
            curr_pos=prev_pos

    color = (255,0,0)
    color2 = (255,255,255)
    if show and correct_marker or (not show and len(markers)>1):
        for marker in markers:
            cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], 
                marker[1] + marker[3]),color2)
        cv2.rectangle(img,(correct_marker[0], correct_marker[1]),(correct_marker[0] + correct_marker[2], 
            correct_marker[1] + correct_marker[3]),color)
        cv2.imshow('frame',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return correct_marker is not None,curr_pos


if __name__ == '__main__':
    folder = sys.argv[1]
    pics = os.listdir(folder) # also sort? 
    d=5
    s=len("image")
    pics = sorted(pics,key=lambda x:int(x[s:s+d]))
    show=False # show picture of detected thingey
    detected_num = 0
    prev_pos=None
    for i,pic in enumerate(pics):
        if pic.endswith(FRAME_EXT):
            image_path = os.path.join(folder,pic)
            found,prev_pos = process_img(image_path,show,prev_pos)
            if found:
                detected_num+=1

            print("found beanie in {} out of {} ({} %)\n".format(detected_num,i+1, 100.0*detected_num/(i+1)))

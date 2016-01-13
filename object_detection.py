from datetime import datetime
import math
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2

#ffprobe ./Test3_Tr1_Session5.MOV -v 0 -select_streams v   -print_format flat -show_entries stream=r_frame_rate
# for tests on 10 Jan 2016
fps= 30000/1001
red_thres=[110,170]
green_thres=[0,60]
sat_thres=[80,240]
FRAME_EXT='png'
MAX_DIST=20 # max dist in pixels between two positions
pixels_to_meters_ratio=(622-519)/3.0 # derived from a known distance in the exported video frame


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
    if show and correct_marker: #or (not show and len(markers)>1):
        for marker in markers:
            cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], 
                marker[1] + marker[3]),color2)
        cv2.rectangle(img,(correct_marker[0], correct_marker[1]),(correct_marker[0] + correct_marker[2], 
            correct_marker[1] + correct_marker[3]),color)
        cv2.imshow('frame',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return correct_marker is not None,curr_pos


def get_speed_timeseries(df,groupby_secs, tracker_turn_on_delay_secs=3):

    # discard data before tracker was turned on
    df.time = df.time - tracker_turn_on_delay_secs
    df = df[df.time>=0]
    # turn into a time series to resample easily
    ts = pd.Series(df.disp.values, df.time)
    ts.index = (ts.index*1e9).to_datetime()
    millis = int(1e3 * groupby_secs)
    # resample & find speed for given groupby_secs interval
    ts = ts.resample(str(millis)+"L",how="max").fillna(method='backfill')
    ts_prev = ts.shift(1)
    speed = (ts - ts_prev) / groupby_secs *(1.0/pixels_to_meters_ratio)
    # make it again to be a seconds offset
    speed.index = map(lambda x: (x.to_datetime() - datetime(1970,1,1,0,0)).total_seconds(), speed.index)
    return speed


if __name__ == '__main__':
    folder = sys.argv[1]
    pics = os.listdir(folder) 
    d=5
    s=len("image")
    pics = sorted(pics,key=lambda x:int(x[s:s+d]))
    show=False # show picture of detected thingey
    detected_num = 0
    prev_pos=None
    time_per_frame = 1.0/fps
    time=0.0
    timeseries=[]
    for i,pic in enumerate(pics):
        if pic.endswith(FRAME_EXT):
            image_path = os.path.join(folder,pic)
            if i>0:
                prev_pos = timeseries[-1]["pos"]
            else:
                prev_pos = None
            found,curr_pos = process_img(image_path,show,prev_pos = prev_pos)
            if found:
                detected_num+=1
                timeseries.append({"time":i*time_per_frame, "pos":curr_pos})
            print("found beanie in {} out of {} ({} %)\n".format(detected_num,i+1, 100.0*detected_num/(i+1)))
        

    df = pd.DataFrame.from_records(timeseries)
    df["disp"] = df.pos.apply(lambda x:x[0])
    speed = get_speed_timeseries(df.copy(),0.5)
    speed.plot()
    plt.show()

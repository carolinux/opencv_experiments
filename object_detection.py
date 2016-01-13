from datetime import datetime
import math
import os
import subprocess
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shutil

# find frame rate 
# ffprobe ./Test3_Tr1_Session5.MOV -v 0 -select_streams v   -print_format flat -show_entries stream=r_frame_rate
# for tests on 10 Jan 2016
fps= 30
red_thres=[110,170]
green_thres=[0,60]
sat_thres=[80,240]
FRAME_BASE='image'
FRAME_EXT='png'
FRAME_DIGITS=5
MAX_DIST=50 # max dist in pixels between two positions
pixels_to_meters_ratio=(622-519)/3.0 # derived from a known distance in the exported video frame


# turn vid to frames
#ffmpeg -i ./Test3_Tr1_Session5.MOV -s hd720 -r 30 -f image2  Test3_Tr1_Session5/image%05d.jpg


def vid_to_frames(vid_path, force_export=False):
    base_folder = os.path.dirname(vid_path)
    frame_dir, ext = os.path.splitext(vid_path)
    if os.path.exists(frame_dir) and os.listdir(frame_dir) and not force_export:
        print("Already have exported frames in {}".format(frame_dir))
    else:
        print("Exporting frames to {}".format(frame_dir))
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)
        fmt= os.path.join(frame_dir,"{}%0{}d.{}".format(FRAME_BASE, FRAME_DIGITS,FRAME_EXT))
        call_args = ["ffmpeg", "-i", str(vid_path),"-s","hd720","-r",str(fps),"-f","image2",fmt]
        print(" ".join(call_args))
        subprocess.call(call_args)

    return frame_dir


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

def show_frame(frame, markers):
    color = (255,0,0)
    color2 = (255,255,255)
    for marker in markers:
        cv2.rectangle(frame,(marker[0], marker[1]),(marker[0] + marker[2], 
            marker[1] + marker[3]),color2)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_frame(img_path,show, prev_pos=None):

    img = cv2.imread(img_path,cv2.IMREAD_COLOR)

    markers = find_marker(img,red_thres=red_thres,green_thres=green_thres, sat_thres=sat_thres)
    correct_marker = None
    if prev_pos is None:
        if len(markers)!=1:
            show_frame(img, markers)
            raise Exception("Can't find initial position of beanie - ambiguous")
        curr_pos = center(markers[0])
        correct_marker = markers[0]

    else:
        found = False
        currd=1000
        for marker in markers:
            d=dist(center(marker),prev_pos)
            if d<MAX_DIST and d<currd:
                curr_pos = center(marker)
                currd=d
                correct_marker = marker
                found=True

        if not found:
            curr_pos=prev_pos

    color = (255,0,0)
    color2 = (255,255,255)
    for marker in markers:
        cv2.rectangle(img,(marker[0], marker[1]),(marker[0] + marker[2], 
            marker[1] + marker[3]),color2)
    if correct_marker:
        cv2.rectangle(img,(correct_marker[0], correct_marker[1]),(correct_marker[0] + correct_marker[2], 
            correct_marker[1] + correct_marker[3]),color)

    if show:
        cv2.imshow('frame',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return correct_marker is not None,curr_pos,img


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
    speed = speed.fillna(0)
    # make it again to be a seconds offset
    speed.index = map(lambda x: (x.to_datetime() - datetime(1970,1,1,0,0)).total_seconds(), speed.index)
    return speed


def process_video(path, force=False):
    frames_folder = vid_to_frames(path)
    parent_folder = os.path.dirname(path)
    basename,_ = os.path.splitext(os.path.basename(path))
    debug_folder = frames_folder+"_debug"
    if not os.path.exists(debug_folder):
        os.mkdir(debug_folder)
    else:
        shutil.rmtree(debug_folder,ignore_errors=True)
        os.mkdir(debug_folder)
    out_speed_filename= os.path.join(parent_folder,basename+"_speed.csv")
    out_speed_graph= os.path.join(parent_folder,basename+"_speed.png")
    out_disp_graph= os.path.join(parent_folder,basename+"_disp.png")
    if os.path.exists(out_speed_filename) and not force:
        print("Video {} has already been processed. Output exists at {}".format(path,out_speed_filename))
        return

    frames = os.listdir(frames_folder) 
    s=len(FRAME_BASE)
    frames = sorted(frames,key=lambda x:int(x[s:s+FRAME_DIGITS]))
    show=False # show picture of detected thingey
    detected_num = 0
    prev_pos=None
    time_per_frame = 1.0/fps
    timeseries=[]
    skip=400
    write_every=15
    for i,frame in enumerate(frames):
        show=False
        if i<skip:
            continue
        if frame.endswith(FRAME_EXT):
            frame_path = os.path.join(frames_folder,frame)
            if i>skip:
                prev_pos = timeseries[-1]["pos"]
            else:
                prev_pos = None
            found,curr_pos,modified_frame = process_frame(frame_path,show,prev_pos = prev_pos)
            if i%write_every==0 or i==skip:
                outfile = os.path.join(debug_folder,"frame{}.png".format(i))
                cv2.imwrite(outfile, modified_frame)
            if found:
                detected_num+=1
                timeseries.append({"time":i*time_per_frame, "pos":curr_pos})
            if i%100==0:
                pct = 100.0*detected_num/(i+1-skip)
                #if pct<80:
                    #show=True
                print("found beanie in {} out of {} ({} %)\n".format(detected_num,i+1-skip, pct))
                print("Time duration: {} to {} seconds".format((i-100)*time_per_frame, i*time_per_frame))
        

    df = pd.DataFrame.from_records(timeseries)
    df["disp"] = df.pos.apply(lambda x:x[0])
    plt.plot(df.time.values, df.disp.values)
    plt.savefig(out_disp_graph)
    plt.clf()
    speed = get_speed_timeseries(df.copy(),0.5)
    speed.to_csv(out_speed_filename)
    speed.plot()
    plt.savefig(out_speed_graph)


def process_all_videos(video_folder):
    vids = os.listdir(video_folder)
    for i,vid in enumerate(vids):
        if not vid.endswith("MOV"):
            continue
        print("Processing video {}".format(vid))
        process_video(os.path.join(video_folder,vid))
        if i>=2:
            return

if __name__ == '__main__':
    process_all_videos(sys.argv[1])

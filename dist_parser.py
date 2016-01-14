from datetime import datetime
import math
import sys

import numpy as np
import pandas as pd

from utils import Filenames as Fn

def read_meta_file(fn):
    recs=[]
    meta={}
    with open(fn,'r') as f:
        first=True
        for line in f:
            line = line.strip()
            if first:
                first=False
                continue
            if line.startswith("#"):
                 continue
            if ":" in line:
                if line[-1]==",":
                    line=line[:-1]
                kv=line.split(":")
                meta[kv[0]]=kv[1].strip()
            else:
                vals = line.split(",")
                try:
                    secs_elapsed = float(vals[0])
                except:
                    print("Could not cast {} to float.".format(vals[0]))
                    secs_elapsed = 0
                location = vals[1].strip()
                rec={"secs_from_prev_to_curr":secs_elapsed,"corner":location}
                recs.append(rec)

    df = pd.DataFrame.from_records(recs)
    return df, meta

def get_meta(meta, keys,func):
    for k in keys:
        if k in meta:
            return func(meta[k])

def get_pitch_length(meta):
    keys=["length","legnth","lenght"]
    return get_meta(meta,keys,float)
    
def get_pitch_width(meta):
    keys=["width","widht","widt"]
    return get_meta(meta,keys,float)

def get_distance(x,distances):
    if pd.isnull(x):
        return 0.0
    if x in distances:
        return distances[x]
    if x[::-1] in distances:
        return distances[x[::-1]]
    return 0.0

def get_corner_to_dist_mapping(w,l,date):
    diag = math.sqrt(l*l + w*w)
    if date==datetime(2016,1,10).date():
        return {"AB":l, "BC":w, "CD":l, "DA":w, "AC":diag, "BD":diag}
    else:
        raise Exception("I don't know how to compute corner to distance mapping for {}".format(date))

def process_file(fn, date):
    df, meta = read_meta_file(fn)
    w = get_pitch_width(meta)
    l = get_pitch_length(meta)
    diag = math.sqrt(l*l + w*w)
    distances=get_corner_to_dist_mapping(w,l,date)
    dfprev = df.shift(1)
    df["route"] = dfprev.corner + df.corner
    df["dist_from_prev_to_curr"] = df.route.apply(lambda x: get_distance(x,distances))
    df["dist_covered"] = df["dist_from_prev_to_curr"].cumsum()
    df["secs_elapsed"] = df["secs_from_prev_to_curr"].cumsum()
    import ipdb; ipdb.set_trace()

def process_all_ground_truth(folder,date):
    all_meta_files = Fn.find_files_with_ext(folder,"meta")
    for fn in all_meta_files:
        process_file(fn,date)

if __name__ == '__main__':
    folder = sys.argv[1]
    date = Fn.extract_date_from_beginning(folder)
    process_all_ground_truth(folder,date)

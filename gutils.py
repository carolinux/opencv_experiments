import os
import sys

from utils import Filenames as Fn

DIST_EXT="dground"
VIDEO_EXT="vground"

def get_matching_ground_truth(session_id, base_folder):
    # session id is something like: Test1_Tr3_Session2
    looking_for = [session_id+"."+DIST_EXT, session_id+"."+VIDEO_EXT]
    ground_truth_files =  [os.path.join(root, name)
             for root, dirs, files in os.walk(base_folder)
             for name in files
             if name.endswith((DIST_EXT,VIDEO_EXT))] 

    res = []
    for fn in ground_truth_files:
        for lfn in looking_for:
            if lfn in fn:
                res.append(fn)

    return res


if __name__ == '__main__':
    match = get_matching_ground_truth(sys.argv[1],sys.argv[2])
    print(match)

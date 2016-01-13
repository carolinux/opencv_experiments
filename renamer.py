

import os
import glob
import sys

from utils import Filenames as Fn


if __name__ == '__main__':

    folder = sys.argv[1]
    if len(sys.argv)>2:
        force=True
    else:
        force=False
    txts = glob.glob(os.path.join(folder,"*txt"))
    metas = glob.glob(os.path.join(folder,"*meta"))
    for txt in txts:
        folder = os.path.dirname(txt)
        basetxt= os.path.basename(txt)
        for meta in metas:
            if "Session"+basetxt in meta:
                newtxtbase,_ = os.path.splitext(meta)
        print basetxt
        print "will rename ", txt," to ",newtxtbase
        if force:
            print "actually renaming"
            Fn.change_filename(txt, newtxtbase)
        

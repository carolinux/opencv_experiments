from datetime import datetime
from glob import glob
import math
import re
import os

from pyproj import Proj,transform


class Files(object):

    @staticmethod
    def filter_lines(f, func):
        """keep only certain lines in file"""

        return filter(lambda line: func(line) ,f.xreadlines())


class GPS(object):

    GNGLL="GNGLL"
    GNMRC="GNRMC"
    GNVTG="GNVTG"
    
    @staticmethod
    def build_datetime_from_hms(date, hms):
        timestr = date.strftime("%Y-%m-%d")+" "+ hms[0:2]+":"+hms[2:4]+":"+"{:8.6f}".format(float(hms[4:]))
        return datetime.strptime(timestr,"%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def extract_date_from_gnmrc(gnmrc_str):
        #"$GNRMC,165025.00,A,4724.14568,N,00835.69659,E,0.085,,021115,,,A*60"
        datestr = gnmrc_str.split(",")[9]
        return datetime.strptime(datestr,"%d%m%y")

    @staticmethod
    def export_nmea(fn, lines):
        # write nmea file for quick visualization
        with open(Filenames.change_ext(Filenames.add_suffix(fn,"_just_gngll"),".nmea"),"w") as out:
            for line in lines:
                out.write(line)

    @staticmethod
    def dm_to_decimal(dm, dir_string, is_lon=False):
        """Converts a DMS coordinate 
        given by GPS data to decimal coordinate.
        Dm is actually of the form [d]ddmm.mmm, no seconds"""
        mult=1
        if is_lon:
            num_digits_for_degree = 3
        else:
            num_digits_for_degree = 2
        if is_lon and dir_string=="W":
            mult = -1
        if not is_lon and dir_string=="S":
            mult = -1

        if dm[-1]==".":
            dm = dm+"0"

        deg = int(dm[:num_digits_for_degree])
        mins = float(dm[num_digits_for_degree:])/60.0

        return mult * (deg + mins)


class Filenames(object):

    @staticmethod
    def add_suffix(filename, suffix):
        base, ext = os.path.splitext(filename)
        return base + suffix + ext

    @staticmethod
    def add_directory_suffix(filename, suffix):
        basename = os.path.basename(filename)
        previous_parents, parent = os.path.split(os.path.dirname(filename))
        new_dir = os.path.join(previous_parents, parent+suffix) 
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return os.path.join(new_dir,basename)

    @staticmethod
    def get_file_list_from_file_or_dir(fn, file_glob=None):
        if not os.path.isdir(fn):
            return [fn]
        if file_glob is None:
            return filter(lambda f: not os.path.isdir(f),glob(os.path.join(fn, "*")))
        else:
            return filter(lambda f: not os.path.isdir(f),glob(os.path.join(fn, file_glob)))


    @staticmethod
    def change_ext(filename, new_ext):
        base, ext = os.path.splitext(filename)
        return base + new_ext


    @staticmethod
    def change_filename(currpath, newfn):
        fn = os.path.basename(currpath)
        par = os.path.dirname(currpath)
        newpath= os.path.join(par,newfn)
        os.rename(currpath,newpath)
        return newpath

    @classmethod
    def is_fused(cls, filename):
        return "fused" in os.path.basename(filename)
    

    @classmethod
    def extract_session_id(cls, filename):
        try:
            if cls.is_fused(filename):
                return re.findall("fused_(.*).txt",filename)[0]
            else:
                return re.findall("session_(.*).txt",filename)[0]
        except:
            return "unknown"


    @staticmethod
    def extract_date_from_beginning(fn,default=None):
        timelen=4 + 2 +2
        basename = os.path.basename(fn.rstrip(os.path.sep))
        try:
            return datetime.strptime(basename[:timelen],"%Y%m%d").date()
        except: 
            if default is not None:
                #print("Could not extract date from filename {}, using provided default.".format(fn))
                return default
            #print("Could not extract date from filename {}, using today's date.".format(fn))
            return datetime.now()




import pandas as pd

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
                rec={"secs_elapsed":secs_elapsed,"corner":location}
                recs.append(rec)

    df = pd.DataFrame.from_records(recs)
    return df, meta
    


fn="/media/carolinux/6662-3462/20160110_data/Test1_Tr1_Session1.txt.meta"
df, meta = read_meta_file(fn)
import ipdb; ipdb.set_trace()

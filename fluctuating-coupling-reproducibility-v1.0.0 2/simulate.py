"""Regenerate a selected saved condition; full runs are computationally expensive."""
import argparse, json
from pathlib import Path
import numpy as np
import numba as nb
from micro import run_one
from special import special_one, meso_one
from controls import pair, one, critical_one

def run(c, replica):
    seed=c["seed0"]+replica
    args=(c["N"],c["K"],c["D"],c["tau"],c["dt"],c["burn"],c["T"],seed)
    kind=c.get("kind","micro")
    if kind=="micro":
        return run_one(c["N"],c["K"],c["D"],c["tau"],c["scheme"],
                       c["dt"],c["burn"],c["T"],round(.1/c["dt"]),seed)
    if kind=="meso": return meso_one(*args)
    if kind in ("correction_a","correction_b","rk4","shared"):
        typ={"correction_a":1,"correction_b":2,"rk4":4,"shared":0}[kind]
        freq= c["seed0"]+10000+(replica//8)*100 if kind=="shared" else -1
        return special_one(*args,typ,freq)
    if kind=="paired":
        return pair(seed,c["scheme"]) if c["N"]==400 else one(seed,c["scheme"])[0]
    if kind=="critical": return critical_one(c["N"],seed)
    raise ValueError(kind)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id",required=True)
    p.add_argument("--replicas",type=int,default=None)
    p.add_argument("--output",default="rerun.npy")
    a=p.parse_args()
    rows=json.loads((Path(__file__).parent/"conditions.json").read_text())
    c=next(c for c in rows if c["id"]==a.id)
    n=c["M"] if a.replicas is None else a.replicas
    if not 1<=n<=c["M"]: p.error("replicas must be within the saved condition")
    out=Path(a.output)
    if out.exists(): p.error("output already exists")
    nb.set_num_threads(1)
    result=np.stack([run(c,m) for m in range(n)]).astype(np.float32)
    np.save(out,result)
    print(out,result.shape)

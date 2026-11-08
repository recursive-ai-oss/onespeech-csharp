"""Only compares numeric oracle/OneSpeech probe files."""
import json
import sys
from pathlib import Path
import numpy as np
reference,ours=map(Path,sys.argv[1:])
for raw in sorted(reference.glob("*.raw.f32")):
    name=raw.name.removesuffix(".raw.f32")
    target=ours/name
    item={"probe":name}
    for kind,a,b in [("raw",raw,target/"raw.f32"),("normalized",reference/(name+".features.f32"),target/"features.f32")]:
        x=np.fromfile(a,dtype="<f4").astype(np.float64); y=np.fromfile(b,dtype="<f4").astype(np.float64)
        if x.shape!=y.shape: raise ValueError("Shape mismatch")
        d=np.abs(x-y); item[kind+"MaxAbs"]=float(d.max()); item[kind+"MeanAbs"]=float(d.mean())
    a,b=reference/(name+".embedding.f32"),target/"embedding.f32"
    if a.exists() and b.exists():
        x=np.fromfile(a,dtype="<f4").astype(np.float64); y=np.fromfile(b,dtype="<f4").astype(np.float64)
        item["cosine"]=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    print(json.dumps(item),flush=True)

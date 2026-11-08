"""Compare fixed probe-set embeddings and captured feature outputs."""
from pathlib import Path
import sys
import json
import numpy as np
ids, ours, reference = sys.argv[1:]
cosines, maxima, means = [], [], []
total_error, total_values = 0.0, 0
for utterance in Path(ids).read_text().splitlines():
    a = np.fromfile(Path(ours)/(utterance+".f32"), dtype="<f4").astype(np.float64)
    b = np.fromfile(Path(reference)/(utterance+".f32"), dtype="<f4").astype(np.float64)
    cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    cosines.append(cosine)
    fa, fb = Path(ours)/(utterance+".features.f32"), Path(reference)/(utterance+".features.f32")
    if fa.exists() and fb.exists():
        diff = np.abs(np.fromfile(fa,dtype="<f4").astype(np.float64)-np.fromfile(fb,dtype="<f4"))
        maxima.append(float(diff.max())); means.append(float(diff.mean()))
        total_error += float(diff.sum()); total_values += int(diff.size)
    print(json.dumps({"id":utterance,"cosine":cosine}),flush=True)
print(json.dumps({"utterances":len(cosines),"minCosine":min(cosines),"meanCosine":float(np.mean(cosines)),"featureFiles":len(maxima),"featureMaxAbs":max(maxima) if maxima else None,"featureMeanAbs":total_error/total_values if total_values else None}),flush=True)

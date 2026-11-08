"""Compare our raw binary outputs to oracle outputs; evaluation only."""
import json
import sys
from pathlib import Path
import numpy as np
reference, ours = sys.argv[1:]
a = np.fromfile(reference + ".features.f32", dtype="<f4")
b = np.fromfile(Path(ours) / "features.f32", dtype="<f4")
if a.shape != b.shape:
    raise SystemExit("Feature shape mismatch")
d = np.abs(a.astype(np.float64) - b)
r = np.fromfile(reference + ".embedding.f32", dtype="<f4").reshape(-1, 512)[0].astype(np.float64)
o = np.fromfile(Path(ours) / "embedding.f32", dtype="<f4").astype(np.float64)
print(json.dumps({"featureMaxAbs": float(d.max()), "featureMeanAbs": float(d.mean()), "cosine": float(np.dot(r,o)/(np.linalg.norm(r)*np.linalg.norm(o)))}))

"""Identifies, from recorded stage tensors, how the released network combines the two fused inputs."""
import sys
import numpy as np

z = np.load(sys.argv[1])
def err(a, b): return float(np.abs(a - b).max())

for name in ["layer3.0.fuse_models.0", "fuse34"]:
    x, y, out = z[name + ":in0"], z[name + ":in1"], z[name + ":out"]
    att = np.tanh(z[name + ".local_att:out"])
    print(name, "att range", att.min(), att.max(), "out range", out.min(), out.max())
    cands = {
        "x*a+y*(1-a)": x * att + y * (1 - att),
        "y*a+x*(1-a)": y * att + x * (1 - att),
        "x+y*a": x + y * att,
        "y+x*a": y + x * att,
        "(x+y)*a": (x + y) * att,
        "x*(1+a)+y*(1-a)": x * (1 + att) + y * (1 - att),
        "x*(1-a)+y*(1+a)": x * (1 - att) + y * (1 + att),
        "x+y": x + y,
        "x*a": x * att,
        "y*a": y * att,
        "2x*a+2y*(1-a)": 2 * x * att + 2 * y * (1 - att),
        "relu(x*a+y*(1-a))": np.maximum(x * att + y * (1 - att),0),
        "relu(y*a+x*(1-a))": np.maximum(y * att + x * (1 - att),0),
        "relu(x+y*a)": np.maximum(x + y * att,0),
        "relu(y+x*a)": np.maximum(y + x * att,0),
        "relu(x*(1+a)+y*(1-a))": np.maximum(x * (1+att) + y * (1-att),0),
        "relu(x*(1-a)+y*(1+a))": np.maximum(x * (1-att) + y * (1+att),0),
    }
    for k, v in cands.items():
        print("  ", k, err(v, out))

# Stage wiring.
print("layer3.0 convs.0 out vs bns.0 out relu:", end=" ")
print("fuse in0 vs relu(bns.0 out)?")
b = z["layer1.0.bns.0:out"]
print("layer1.0.convs.1 in0 vs bn0 relu:", err(z["layer1.0.convs.1:in0"], np.maximum(b, 0)),
      "vs bn0:", err(z["layer1.0.convs.1:in0"], b))
print("layer1.0.conv3 in0 (concat) first half vs relu(bns.0):", err(z["layer1.0.conv3:in0"][:, :26], np.maximum(b, 0)))
b1 = z["layer1.0.bns.1:out"]
print("second half vs relu(bns.1):", err(z["layer1.0.conv3:in0"][:, 26:], np.maximum(b1, 0)), "vs bns.1", err(z["layer1.0.conv3:in0"][:, 26:], b1))
x = z["layer1.0.conv1:out"]
print("layer1.0 convs.0 in0 vs relu(bn1(conv1)) first half", err(z["layer1.0.convs.0:in0"], np.maximum(z["layer1.0.bn1:out"][:, :26], 0)))
print("bn1 in0 vs conv1 out", err(z["layer1.0.bn1:in0"], x))
y0 = np.maximum(b, 0)
x1 = np.maximum(z["layer1.0.bn1:out"][:, 26:], 0)
print("convs.1 in0 vs y0+x1:", err(z["layer1.0.convs.1:in0"], y0 + x1))
# block output
o = z["layer1.0:out"]
pre = z["layer1.0.bn3:out"] + z["layer1.0.shortcut:out"]
print("block out vs relu(bn3+shortcut):", err(o, np.maximum(pre, 0)))
print("layer3 fuse in0 vs relu(bns.0)?  need bn tensors; layer3.0 convs.1 in0 vs fuse out:", err(z["layer3.0.convs.1:in0"], z["layer3.0.fuse_models.0:out"]))
print("fuse34 in0 vs layer3_ds out:", err(z["fuse34:in0"], z["layer3_ds:out"]), "in1 vs layer4 out:", err(z["fuse34:in1"], z["layer4:out"]))
print("fuse34 in0 vs layer4 out:", err(z["fuse34:in0"], z["layer4:out"]))
print("seg_1 in0 shape", z["seg_1:in0"].shape, "fuse34 out", z["fuse34:out"].shape)
f = z["fuse34:out"]
b_, c, h, w = f.shape
flat = f.reshape(b_, c * h, w)
mean = flat.mean(2); std = flat.std(2, ddof=1)
cand = np.concatenate([mean, std], 1)
print("pool mean+std(unbiased) vs seg in:", err(cand, z["seg_1:in0"]))
std0 = flat.std(2, ddof=0)
print("pool mean+std(biased):", err(np.concatenate([mean, std0], 1), z["seg_1:in0"]))


for blk in ["layer1.0","layer3.0"]:
    bn3=z[blk+".bn3:out"]; sc=z[blk+".shortcut:out"]; o=z[blk+":out"]
    for k,v in {"relu(bn3+sc)":np.maximum(bn3+sc,0),"relu(bn3)+sc":np.maximum(bn3,0)+sc}.items(): print(blk,k,err(v,o))
b1=np.maximum(z["layer3.0.bn1:out"],0); h=b1.shape[1]//2
f="layer3.0.fuse_models.0"
y0=np.maximum(z["layer3.0.bns.0:out"],0)
print("fuse in0 vs y0",err(z[f+":in0"],y0),"in1 vs x1",err(z[f+":in1"],b1[:,h:]),"in0 vs x1",err(z[f+":in0"],b1[:,h:]),"in1 vs y0",err(z[f+":in1"],y0))
y1=np.maximum(z["layer3.0.bns.1:out"],0)
print("concat to conv3 is [y0,y1]:",err(z["layer3.0.conv3:in0"],np.concatenate([y0,y1],1)))
print("layer3.0 conv1 in0 is layer2 out; layer4.0 out vs fuse34 in0",err(z["layer4:out"],z["fuse34:in0"]))
print("layer3 stride: conv1 in/out",z["layer3.0.conv1:in0"].shape,z["layer3.0.conv1:out"].shape)

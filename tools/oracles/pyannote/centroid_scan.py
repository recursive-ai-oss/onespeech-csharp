import numpy as np, glob, sys
d = sys.argv[1]
seg = np.load(glob.glob(d + '/*-segmentation.npy')[0])
emb = np.load(sorted(glob.glob(d + '/*-embeddings.npy'))[-1])
cent = np.load(d + '/speaker_embeddings.npy')
C = seg.shape[0]
b = seg > 0.5
on = b.sum(2)  # [C,589]
clean = np.stack([(b[:, :, s] & (on == 1)).sum(1) for s in range(3)], 1)  # [C,3]
active = b.sum(1)  # [C,3]
E = emb.reshape(C * 3, 256)
ok = np.isfinite(E).all(1)
print('chunks', C, 'finite emb', ok.sum(), 'nan', (~ok).sum(), 'active>0', (active.reshape(-1) > 0).sum())
cl = clean.reshape(-1); ac = active.reshape(-1)
print('finite but zero active', (ok & (ac == 0)).sum(), 'nonfinite but active', ((~ok) & (ac > 0)).sum(), 'nonfinite with clean>0', ((~ok) & (cl > 0)).sum())
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
U = np.zeros_like(E); U[ok] = unit(E[ok])
S = U @ unit(cent).T  # [n,4]
lab = S.argmax(1)
print('cluster sizes by nearest ref centroid (finite):', [(int(((lab == k) & ok).sum())) for k in range(len(cent))])
for t in [0, 1, 20, 40, 59, 90, 120, 150, 200, 300, 400]:
    out = []
    for k in range(len(cent)):
        m = ok & (lab == k) & (cl >= t)
        if m.sum() == 0: out.append('-'); continue
        mean = E[m].mean(0); meanu = U[m].mean(0)
        out.append('%.7f/%.7f(%d)' % (unit(mean) @ unit(cent[k]), unit(meanu) @ unit(cent[k]), m.sum()))
    print('clean>=%d' % t, ' '.join(out))
# active-based
for t in [0, 59, 118, 177, 295]:
    out = []
    for k in range(len(cent)):
        m = ok & (lab == k) & (ac >= t)
        if m.sum() == 0: out.append('-'); continue
        out.append('%.7f(%d)' % (unit(E[m].mean(0)) @ unit(cent[k]), m.sum()))
    print('active>=%d' % t, ' '.join(out))
# norm hint
print('ref centroid norms', np.linalg.norm(cent, axis=1), 'mean emb norms', [np.linalg.norm(E[ok & (lab == k)].mean(0)) for k in range(len(cent))])
# Fine scan around the best threshold, on raw-embedding means.
print('fine scan (clean frames; ratio of 589 frames in brackets)')
for t in range(100, 141, 2):
    out = []
    for k in range(len(cent)):
        m = ok & (lab == k) & (cl >= t)
        out.append('%.8f' % (unit(E[m].mean(0)) @ unit(cent[k])))
    print('clean>=%d (%.4f)' % (t, t / 589), ' '.join(out))
